#include "gplib.hpp"
#include <nlopt.hpp>
#include <ctime>

using namespace arma;
using namespace std;

namespace gplib {

  struct gp_reg_multi::implementation {
    shared_ptr<multioutput_kernel_class> kernel;
    vector<mat> X;
    vector<vec> y;
    vector<mat> M;
    double sigma = 0.01;
    size_t state = FULL;

    vec eval_mean(vector<mat> &data) {
      size_t total_size = 0;
      for (size_t i = 0; i < data.size(); ++i) {
        total_size += data[i].n_rows;
      }
      return zeros<vec> (total_size);
    }

    mv_gauss predict(const vector<mat> &new_data) {
      //Add new data to observations
      vector<mat> M(X.size());
      vec fill_y;
      size_t total_rows = 0;
      for (size_t i = 0; i < X.size(); i++) {
        M[i] = join_vert (X[i], new_data[i]);
        fill_y = join_cols<mat> (fill_y, y[i]);
        fill_y = join_cols<mat> (fill_y, zeros<vec>(new_data[i].n_rows));
        total_rows += M[i].n_rows;
      }

      //Compute Covariance
      mat cov = kernel -> eval(M, M);
      //Set mean
      vec mean = eval_mean(M);
      //Set alredy observed Values
      vector<bool> observed(mean.n_rows, false);
      size_t start = 0;
      for (size_t i = 0; i < M.size(); i++) {
        for (size_t j = 0; j < X[i].n_rows; j++)
          observed[start + j] = true;
        start += M[i].n_rows;
      }
      //Conditon Multivariate Gaussian
      mv_gauss gd(mean, cov);
      return gd.conditional(fill_y, observed);
    }

    mat comp_Q(const vector<mat> &a, const vector<mat> &b, vector<mat> &u) {
      mat kuui =  kernel-> eval(u, u).i();
      return kernel-> eval(a, u) * kuui * kernel-> eval(u, b);
    }

    mv_gauss predict_FITC(const vector<mat> &new_x) {
      mat Qn = comp_Q(X, X, M);
      mat Qm = comp_Q(new_x, new_x, M);
      mat Kff_diag = kernel-> eval(X, X, true);
      mat lambda = Kff_diag - diagmat(Qn);
      mat Kuu = kernel-> eval(M, M);
      mat Kfu = kernel-> eval(X, M);
      mat Kuf = kernel-> eval(M, X);
      mat Knn = kernel-> eval(new_x, new_x);
      mat Kun = kernel-> eval(M, new_x);
      mat Knu = kernel-> eval(new_x, M);
      lambda = (lambda + sigma * eye(lambda.n_rows, lambda.n_cols)).i();
      mat E = (Kuu + Kuf * lambda * Kfu).i();
      mat Y = flatten(y);
      mat mean = Knu * E * Kuf * lambda * Y;
      mat cov = Knn - Qm + Knu * E * Kun;

      return mv_gauss(mean, cov);
    }

    mv_gauss marginal() {
      vec mean = eval_mean(X);
      mat cov = kernel-> eval(X, X);
      return mv_gauss(mean, cov);
    }

    void set_params(const vector<double> &params) {
      size_t M_size = 0;
      sigma = params.back();
      for(size_t i = 0; i < M.size(); ++i) {
        M_size += M[i].size();
      }
      vector<double> kernel_params(params.size() - M_size - 1), M_params(M_size);
      split(params, kernel_params, M_params);
      kernel-> set_params(kernel_params);
      M = unflatten(M_params, M);
    }

    vector<double> get_params() {
      vector<double> params = kernel-> get_params();
      vector<double> flatten_M = flatten(M);
      params.insert(params.end(), flatten_M.begin(), flatten_M.end());
      params.push_back(sigma);
      return params;
    }

    double log_marginal() {
      return marginal().log_density(flatten(y));
    }

    double log_marginal_fitc() {
      size_t N = 0;
      for (size_t i = 0; i < X.size(); ++i)
        N += X[i].n_rows;

      mat Qff = comp_Q (X, X, M);
      Qff = force_symmetric(Qff);
      mat Kff_diag = kernel-> eval(X, X, true);

      mat lambda = Kff_diag - diagmat(Qff) + sigma *
        eye<mat> (Qff.n_rows, Qff.n_cols);
      mat B = force_diag(Qff + lambda);
      B = chol(B);
      double log_det = 0;
      for (size_t i = 0; i < Qff.n_rows; ++i)
        log_det += log(B(i, i));
      double ans = -log_det;
      mat flat_y = flatten (y);
      mat tmp = (flat_y.t() * (Qff + lambda).i() * flat_y);
      ans -= 0.5 * tmp(0,0);
      ans -= 0.5 * N * log (2.0 * pi);
      return ans;
    }

    static double training_obj(const vector<double> &theta,
        vector<double> &grad, void *fdata) {

      implementation *pimpl = (implementation*) fdata;
      pimpl-> kernel-> set_params(theta);

      double ans = pimpl-> log_marginal();

      vec mx = pimpl-> eval_mean(pimpl-> X);
      mat K = pimpl-> kernel-> eval(pimpl-> X, pimpl-> X);
      mat Kinv = K.i();
      vec diff = flatten(pimpl-> y);
      mat dLLdK = -0.5 * Kinv + 0.5 * Kinv * diff * diff.t() * Kinv;

      for (size_t d = 0; d < grad.size(); d++) {
        mat dKdT = pimpl-> kernel-> derivate(d, pimpl-> X, pimpl-> X);
        grad[d] = trace(dLLdK * dKdT);
      }

      return ans;
    }

    static double training_obj_FITC(const vector<double> &theta,
      vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;
      pimpl-> set_params(theta);

      double ans = pimpl-> log_marginal_fitc();

      mat flat_y = flatten (pimpl-> y);
      mat Qff = force_symmetric(
                pimpl-> comp_Q (pimpl-> X, pimpl-> X, pimpl-> M));

      mat I = eye<mat> (Qff.n_rows, Qff.n_cols);
      mat Kff_diag = pimpl-> kernel-> eval(pimpl-> X, pimpl-> X, true);
      mat lambda = Kff_diag - diagmat (Qff) + pimpl-> sigma * I;
      mat Ri = (Qff + lambda).i();
      mat ytRi = flat_y.t() * Ri;
      mat Riy = Ri * flat_y;
      mat Kuui = (pimpl-> kernel-> eval (pimpl-> M, pimpl-> M)).i();
      mat Kuf = pimpl-> kernel-> eval(pimpl-> M, pimpl-> X);
      mat KuuiKuf = Kuui * Kuf;
      mat Kfu = pimpl-> kernel-> eval(pimpl-> X, pimpl-> M);
      mat KfuKuui = Kfu * Kuui;

      const vector<double> &lb = pimpl-> kernel-> get_lower_bounds();
      const vector<double> &ub = pimpl-> kernel-> get_upper_bounds();

      for (size_t d = 0; d < grad.size(); d++) {
        if (d < lb.size() && ub[d] <= lb[d]) {
           grad[d] = 0;
           continue;
        }
        mat dRdT;
        if(d + 1 < grad.size()) {
          mat dKfudT = pimpl-> kernel-> derivate (d, pimpl-> X, pimpl-> M);
          mat dKuudT = pimpl-> kernel-> derivate (d, pimpl-> M, pimpl-> M);
          mat dKufdT = pimpl-> kernel-> derivate (d, pimpl-> M, pimpl-> X);
          mat dQffdT = KfuKuui * (dKufdT - dKuudT * KuuiKuf) + dKfudT * KuuiKuf;
          mat dKffdT_diag;
          //If it is one of the pseudo-inputs dKff should be 0
          if (d > pimpl-> kernel-> n_params())
            dKffdT_diag = zeros<mat>(Qff.n_rows, Qff.n_cols);
          else
            dKffdT_diag = pimpl-> kernel-> derivate (d, pimpl-> X, pimpl-> X, true);
          dRdT = dQffdT + dKffdT_diag - diagmat(dQffdT);

        } else { // Special case for sigma.
          dRdT = 2 * sqrt(pimpl-> sigma) * I;
        }

        double t = accu(Ri % dRdT); // trace(Ri * dRdT);
        mat ans = -t + ytRi * dRdT * Riy;

        grad[d]  = 0.5 * ans(0,0);
        if (d < pimpl-> kernel-> get_kernels().size() * pimpl-> X.size() * pimpl-> X.size()) {
          grad[d] *= 2;
        }
      }
      return ans;
    }


    double train(int max_iter, double tol) {
      nlopt::opt best(nlopt::LD_MMA, kernel-> n_params());
      best.set_max_objective(implementation::training_obj, this);
      best.set_xtol_rel(tol);
      best.set_maxeval(max_iter);

      best.set_lower_bounds(kernel-> get_lower_bounds());
      best.set_upper_bounds(kernel-> get_upper_bounds());

      double error; //final value of error function
      vector<double> x = kernel-> get_params();
      best.optimize(x, error);
      kernel-> set_params(x);
      return error;
    }

    double train_FITC(int max_iter, double tol) {
      size_t M_size = 0;
      for (size_t i = 0; i < M.size(); ++i)
        M_size += M[i].size();
      size_t n_params = kernel-> n_params() + M_size + 1; // added sigma
      nlopt::opt best(nlopt::LD_MMA, n_params);
      best.set_max_objective(implementation::training_obj_FITC, this);
      best.set_xtol_rel(tol);
      best.set_maxeval(max_iter);
      vector<double> lb = kernel-> get_lower_bounds();
      vector<double> ub = kernel-> get_upper_bounds();

      lb.resize(lb.size() + M_size, -HUGE_VAL);
      ub.resize(ub.size() + M_size, HUGE_VAL);
      lb.push_back(0.0);
      ub.push_back(HUGE_VAL);
      best.set_lower_bounds(lb);
      best.set_upper_bounds(ub);

      double error; //final value of error function
      vector<double> x = get_params();
      best.optimize(x, error);
      set_params(x);

      return error;
    }

  };

  gp_reg_multi::gp_reg_multi() {
    pimpl = new implementation();
  }

  gp_reg_multi::~gp_reg_multi() {
    delete pimpl;
  }

  void gp_reg_multi::set_kernel(const shared_ptr<multioutput_kernel_class> &k) {
    pimpl-> kernel = k;
  }

  void gp_reg_multi::set_training_set(const vector<mat> &X,
      const vector<vec> &y) {

    pimpl-> X = X;
    pimpl-> y = y;
  }

  double gp_reg_multi::train(const int max_iter, const double tol) {
    return pimpl-> train(max_iter, tol);
  }

  double gp_reg_multi::train(const int max_iter, const double tol,
    const size_t num_pi){
    //Initial check
    if (pimpl-> X.size() == 0 || pimpl-> X[0].size() == 0)
      throw logic_error("Parameters Uninitialized");

    pimpl-> M = vector<mat>(pimpl-> X.size());
    for (size_t i = 0; i < pimpl-> X.size(); ++i){
      //Check for too many inducing points
      if (num_pi > pimpl-> X[i].n_rows)
        throw length_error("Too many inducing points");
      //Initial matrix
      pimpl-> M[i] = zeros<mat>(num_pi, pimpl-> X[i].n_cols);
      //Fill matrix
      for (size_t j = 0; j < pimpl-> X[i].n_cols; ++j){
        double col_max = pimpl-> X[i].col(j).max();
        double col_min = pimpl-> X[i].col(j).min();
        double step = (col_max - col_min) / num_pi;
        double cur = col_min;
        for (size_t k = 0; k < num_pi; ++k){
          pimpl-> M[i](k, j) = cur;
          cur += step;
        }
      }
    }
    pimpl-> state = FITC;
    return pimpl-> train_FITC(max_iter, tol);
  }

  double gp_reg_multi::train(const int max_iter, const double tol,
    const vector<size_t> num_pi){
    //Initial check
    if (pimpl-> X.size() == 0 || pimpl-> X[0].size() == 0)
      throw logic_error("Parameters Uninitialized");
    if (num_pi.size() != pimpl-> X.size())
      throw length_error("Wrong inducing point vector size");

    pimpl-> M = vector<mat>(pimpl-> X.size());
    for (size_t i = 0; i < pimpl-> X.size(); ++i){
      //Check for too many inducing points
      if (num_pi[i] >= pimpl-> X[i].n_rows)
        throw length_error("Too many inducing points");
      //Create initial matrix
      pimpl-> M[i] = zeros<mat>(num_pi[i], pimpl-> X[i].n_cols);
      //Fill matrix
      for (size_t j = 0; j < pimpl-> X[i].n_cols; ++j){
        double col_max = pimpl-> X[i].col(j).max();
        double col_min = pimpl-> X[i].col(j).min();
        double step = (col_max - col_min) / num_pi[i];
        double cur = col_min;
        for (size_t k = 0; k < num_pi[i]; ++k){
          pimpl-> M[i](k, j) = cur;
          cur += step;
        }
      }
    }
    pimpl-> state = FITC;
    return pimpl-> train_FITC(max_iter, tol);
  }

  mv_gauss gp_reg_multi::full_predict(const vector<mat> &new_data) {
    if (pimpl-> state == FITC)
      return pimpl-> predict_FITC(new_data);
    else
      return pimpl-> predict(new_data);
  }

  arma::vec gp_reg_multi::predict(const vector<arma::mat> &new_data) const {
    mv_gauss g;
    if (pimpl-> state == FITC)
      g = pimpl-> predict_FITC(new_data);
    else
      g = pimpl-> predict(new_data);
    return g.get_mean();
  }

  vector<double> gp_reg_multi::get_params() const {
    return pimpl-> get_params();
  }

  void gp_reg_multi::set_params(const vector<double> &params) {
    pimpl-> set_params(params);
  }
};
