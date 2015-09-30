#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib {

  struct gp_reg_multi::implementation {
    shared_ptr<multioutput_kernel_class> kernel;
    vector<mat> X;
    vector<vec> y;

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

    mv_gauss marginal() {
      vec mean = eval_mean(X);
      mat cov = kernel->eval(X, X);
      return mv_gauss(mean, cov);
    }

    vec flatten(vector<vec> &y) {
      vec flat;
      for (size_t i = 0; i < y.size(); i++) {
        flat = join_cols<mat> (flat, y[i]);
      }
      return flat;
    }

    double log_marginal() {
      return marginal().log_density(flatten(y));
    }

    static double training_obj(const vector<double> &theta, vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;
      pimpl->kernel->set_params(theta);
      double ans = pimpl-> log_marginal();

      vec mx = pimpl->eval_mean(pimpl-> X);
      mat K = pimpl->kernel->eval(pimpl-> X, pimpl-> X);
      mat Kinv = K.i();
      vec diff = pimpl->flatten(pimpl-> y);
      mat dLLdK = -0.5 * Kinv + 0.5 * Kinv * diff * diff.t() * Kinv;
      for (size_t d = 0; d < grad.size(); d++) {
        mat dKdT = pimpl-> kernel-> derivate(d, pimpl-> X, pimpl-> X);
        grad[d] = trace(dLLdK * dKdT);
      }
      return ans;
    }

    double train(int max_iter) {
      nlopt::opt my_min(nlopt::LD_MMA, kernel->n_params());
      my_min.set_max_objective(implementation::training_obj, this);
      my_min.set_xtol_rel(1e-4);
      my_min.set_maxeval(max_iter);

      my_min.set_lower_bounds(kernel-> get_lower_bounds());
      my_min.set_upper_bounds(kernel-> get_upper_bounds());

      double error; //final value of error function
      vector<double> x = kernel-> get_params();
      my_min.optimize(x, error);
      kernel->set_params(x);
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
    pimpl->kernel = k;
  }

  void gp_reg_multi::set_training_set(const vector<mat> &X, const vector<vec> &y) {
    pimpl->X = X;
    pimpl->y = y;
  }

  double gp_reg_multi::train(const int max_iter) {
    return pimpl->train(max_iter);
  }

  mv_gauss gp_reg_multi::full_predict(const vector<mat> &new_data) {
    return pimpl->predict(new_data);
  }

  arma::vec gp_reg_multi::predict(const vector<arma::mat> &new_data) const {
    mv_gauss g = pimpl->predict(new_data);
    return g.get_mean();
  }
};
