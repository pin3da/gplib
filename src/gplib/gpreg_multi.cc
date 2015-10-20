#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib {

  struct gp_reg_multi::implementation {
    shared_ptr<multioutput_kernel_class> kernel;
    vector<mat> X;
    vector<vec> y;
    vector<mat> M;
    double sigma;
    size_t state;

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

    mat comp_Q(vector<mat> &a, vector<mat> &b, vector<mat> &u) {
      mat kuu =  kernel-> eval(u, u).i();
      return kernel-> eval(a, u) * kuu * kernel-> eval(u, b);
    }

    mv_gauss predict_FITC(const vector<mat> &new_x) {
      // TODO: optimize sigma.
      sigma = 0.01;
      mat Qn = comp_Q(X, X, M);
      mat lambda = diagmat(kernel-> eval(X, X) - Qn);
      mat tmp = (lambda + sigma * eye(lambda.n_rows, lambda.n_cols)).i();
      mat B = kernel-> eval(M, M) * kernel-> eval(M, X) * tmp *
              kernel-> eval(X, M);
      mat Y = flatten(y);
      mat mean = kernel-> eval(new_x, M) * B.i() * kernel-> eval(M, X) *
                 tmp * Y;
      tmp     = kernel-> eval(M, M).i() - B.i();
      mat cov = kernel-> eval(new_x, new_x) - kernel-> eval(new_x, M) *
                tmp * kernel-> eval(M, new_x) + sigma;
      return mv_gauss(mean, cov);
    }


    mv_gauss marginal() {
      vec mean = eval_mean(X);
      mat cov = kernel-> eval(X, X);
      return mv_gauss(mean, cov);
    }

    vec flatten(vector<vec> &y) {
      vec flat;
      for (size_t i = 0; i < y.size(); i++) {
        flat = join_cols<mat> (flat, y[i]);
      }
      return flat;
    }

    vector<double> flatten(vector<mat> &M) {
      size_t t_size = M.size() * M[0].size();
      vector<double> ans(t_size);
      size_t iter = 0;
      for (size_t q = 0; q < M.size(); ++q) {
        copy (M[q].begin(), M[q].end(), ans.begin() + iter);
        iter += M[q].size();
      }
      return ans;
    }

    void unflatten(vector<double> &M_params) {
      size_t iter = 0;
      for(size_t q = 0; q < M.size(); ++q)
        for(size_t i = 0; i < M[q].n_rows; ++i)
          for(size_t j = 0; j < M[q].n_cols; ++j) {
            M[q](i, j) = M_params[iter];
            ++iter;
          }
    }

    void split(const vector<double> &theta, vector<double> &kernel_params, vector<double> &M_params) {
      copy(theta.begin(), theta.begin() + kernel_params.size(), kernel_params.begin());
      copy(theta.begin() + kernel_params.size() + 1, theta.end(), M_params.begin());
    }

    double log_marginal() {
      return marginal().log_density(flatten(y));
    }

    double log_marginal_fitc() {
      mat Qn = comp_Q(X, X, M);
      mat gamma = diagmat(kernel-> eval(X, X) - Qn) + sigma * eye<mat>(Qn.n_rows, Qn.n_cols);
      gamma     /= sigma;

      size_t total_N = 0, total_M = 0;
      for (size_t  i = 0; i < X.size(); ++i) {
        total_N += X[i].n_rows;
        total_M += M[i].n_rows;
      }

      mat Kmm =  kernel-> eval(M, M);
      mat A  = sigma * Kmm + kernel-> eval(M, X) * (gamma.i()) * kernel-> eval(X, M);
      double L1 = log(det(A)) - log(det(Kmm)) + log(det(gamma)) + (total_N - total_M) * log(sigma);

      mat y_sub   = sqrt(gamma).i() * flatten(y);
      mat Kmn_sub = (sqrt(gamma).i() * kernel-> eval(X, M)).t();

      double ny  = norm(y_sub);
      double tmp = norm(sqrt(A).i() * Kmn_sub * y_sub);
      double L2  = (ny * ny - tmp * tmp) / sigma;

      return L1 + L2 + total_N * log(2 * pi);
    }

    static double training_obj(const vector<double> &theta, vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;
      pimpl-> kernel-> set_params(theta);
      double ans = pimpl-> log_marginal();

      vec mx = pimpl-> eval_mean(pimpl-> X);
      mat K = pimpl-> kernel-> eval(pimpl-> X, pimpl-> X);
      mat Kinv = K.i();
      vec diff = pimpl-> flatten(pimpl-> y);
      mat dLLdK = -0.5 * Kinv + 0.5 * Kinv * diff * diff.t() * Kinv;
      for (size_t d = 0; d < grad.size(); d++) {
        mat dKdT = pimpl-> kernel-> derivate(d, pimpl-> X, pimpl-> X);
        grad[d] = trace(dLLdK * dKdT);
      }
      return ans;
    }


    double derivate_l1(const mat &A, const mat &A_dot, const mat &Kmm,
        const mat &Km_dot, const mat &gamma_2sub_dot) {

      mat s_A    = sqrt(A);
      mat s_A_t  = s_A.t();
      mat s_Km   = sqrt(Kmm);
      mat s_Km_t = s_Km.t();

      double ans = trace(s_A * A_dot * s_A_t) - trace(s_Km * Km_dot * s_Km_t) +
                   trace(gamma_2sub_dot);
      return ans * 0.5;

    }

    double derivate_l2(const double &sigma, const mat &y_sub,
        const mat &gamma_2sub_dot, const mat &A, const mat &A_dot,
        const mat &Kmn_sub, const mat &Kmn_sub_dot) {


      mat y_sub_t = y_sub.t();
      mat s_A     = sqrt(A);
      mat s_A_t   = s_A.t();

      mat tmp   = s_A * Kmn_sub * y_sub;
      mat tmp_t = tmp.t();
      double ans = 0.5 * y_sub_t * gamma_2sub_dot * y_sub +
        tmp_t * (0.5 * s_A * A_dot * s_A_t) * tmp -
        s_A * Kmn_sub_dot * y_sub +
        s_A * Kmn_sub * gamma_2sub_dot * y_sub;

      return ans / sigma;

    }

    static double training_obj_FITC(const vector<double> &theta, vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;

      // TODO: implement set_params for gpreg_multi and move the
      // following lines there. (We need to set sigma there too).
      size_t M_size = pimpl-> M.size() * pimpl-> M[0].size();
      vector<double> kernel_params(theta.size() - M_size), M_params(M_size);
      pimpl-> split(theta, kernel_params, M_params);
      pimpl-> kernel-> set_params(kernel_params);
      pimpl-> unflatten(M_params);

      double ans = pimpl-> log_marginal_fitc();

      mat Qn = pimpl-> comp_Q(pimpl-> X, pimpl-> X, pimpl-> M);
      mat gamma = diagmat(pimpl-> kernel-> eval(pimpl-> X, pimpl-> X) - Qn) +
                  pimpl-> sigma * eye<mat>(Qn.n_rows, Qn.n_cols);

      gamma /= pimpl-> sigma;

      mat sqrt_gamma = sqrt(gamma);
      mat gamma_i = gamma.i();
      mat Kmm   = pimpl-> kernel-> eval(pimpl-> M, pimpl-> M);
      mat Kmm_i = Kmm.i();
      mat Knm   = pimpl-> kernel-> eval(pimpl-> X, pimpl-> M);
      mat Kmn   = Knm.i();
      mat Knm_sub = sqrt_gamma * Knm;
      mat Kmn_sub = sqrt_gamma * Kmn;
      mat y_sub   = sqrt(gamma).i() * pimpl-> flatten(pimpl-> y);

      double _s = pimpl-> sigma;
      mat A = _s * Kmm + Kmn * gamma_i * Knm;


      for (size_t d = 0; d < grad.size(); d++) {
        mat dKmmdT = pimpl-> kernel-> derivate(d, pimpl-> M, pimpl-> M);
        mat dKmndT = pimpl-> kernel-> derivate(d, pimpl-> M, pimpl-> X);
        mat dKnmdT = pimpl-> kernel-> derivate(d, pimpl-> X, pimpl-> M);
        mat dKnndT = pimpl-> kernel-> derivate(d, pimpl-> X, pimpl-> X);

        mat Knm_sub_dot = sqrt_gamma * dKnmdT;
        mat Kmn_sub_dot = sqrt_gamma * dKmndT;
        mat Knn_sub_dot = sqrt_gamma * dKnndT;

        mat gamma_2sub_dot = diagmat(Knn_sub_dot - 2 * Knm_sub_dot * Kmm_i * Kmn_sub +
                             Knm_sub * Kmm_i * dKmmdT * Kmm_i * Kmn_sub);
            gamma_2sub_dot /= _s;


        mat A_dot = _s * dKmmdT + 2 * symmatl(Kmn_sub_dot * Knm_sub) -
                    Kmn_sub * gamma_2sub_dot * Knm_sub;

        grad[d] = pimpl-> derivate_l1(A, A_dot, Kmm, dKmmdT, gamma_2sub_dot)
                  + pimpl->derivate_l2(_s, y_sub, gamma_2sub_dot, A, A_dot,
                                       Kmn_sub, Kmn_sub_dot);
      }
      return ans;
    }


    double train(int max_iter) {
      nlopt::opt my_min(nlopt::LD_MMA, kernel-> n_params());
      my_min.set_max_objective(implementation::training_obj, this);
      my_min.set_xtol_rel(1e-4);
      my_min.set_maxeval(max_iter);

      my_min.set_lower_bounds(kernel-> get_lower_bounds());
      my_min.set_upper_bounds(kernel-> get_upper_bounds());

      double error; //final value of error function
      vector<double> x = kernel-> get_params();
      my_min.optimize(x, error);
      kernel-> set_params(x);
      return error;
    }

    double train_FITC(int max_iter) {
      size_t M_size = M.size() * M[0].size();
      size_t n_params = kernel-> n_params() + M_size;
      nlopt::opt my_min(nlopt::LD_MMA, n_params);
      my_min.set_max_objective(implementation::training_obj_FITC, this);
      my_min.set_xtol_rel(1e-4);
      my_min.set_maxeval(max_iter);
      vector<double> lb = kernel-> get_lower_bounds();
      vector<double> ub = kernel-> get_upper_bounds();
      lb.resize(lb.size() + M_size, -HUGE_VAL);
      ub.resize(ub.size() + M_size, HUGE_VAL);
      my_min.set_lower_bounds(lb);
      my_min.set_upper_bounds(ub);

      double error; //final value of error function
      vector<double> x = kernel-> get_params();
      vector<double> flatten_M = flatten(M);
      x.insert(x.end(), flatten_M.begin(), flatten_M.end());

      my_min.optimize(x, error);
      vector<double> kernel_params(x.size() - M_size), M_params(M_size);
      split(x, kernel_params, M_params);
      kernel-> set_params(kernel_params);
      unflatten(M_params);

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

  void gp_reg_multi::set_training_set(const vector<mat> &X, const vector<vec> &y) {
    pimpl-> X = X;
    pimpl-> y = y;
  }

  double gp_reg_multi::train(const int max_iter, const size_t type, void *param) {
    pimpl-> state = type;
    if (type == FITC) {
      size_t num_pi = *((size_t *) param);
      pimpl-> M = vector<mat> (pimpl-> X.size(),
                  zeros<mat>(num_pi, pimpl-> X[0].n_cols));

      return pimpl-> train_FITC(max_iter);
    } else
      return pimpl-> train(max_iter);
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
};
