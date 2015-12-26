#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib {

  struct gp_reg::implementation {
    shared_ptr<kernel_class> kernel;
    mat X; //Matrix of inputs
    vec y; //vector of outputs
    // double noise;

    vec eval_mean(const arma::mat& data) {
      // For the moment just use the zero mean
      return zeros<vec>(data.n_rows);
    }

    mv_gauss predict(const arma::mat& new_data) {
      mat M = join_vert(X, new_data);
      int n = X.n_rows, n_val = new_data.n_rows;
      mat cov = kernel-> eval(M, M);
      vec mean = eval_mean(M);
      mv_gauss gd(mean, cov);
      vector<bool> observed(n + n_val, false);
      for (int i = 0; i < n; ++i)
        observed[i] = true;
      return gd.conditional(y, observed);
    }

    mv_gauss marginal() {
      vec mean = eval_mean(X);
      mat cov = kernel-> eval(X, X);
      return mv_gauss(mean, cov);
    }

    double log_marginal() {
      return marginal().log_density(y);
    }

    static double training_obj(const vector<double> &theta, vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;
      pimpl-> kernel-> set_params(theta);
      double ans = pimpl-> log_marginal();

      vec mx = pimpl-> eval_mean(pimpl-> X);
      mat K = pimpl-> kernel-> eval(pimpl-> X, pimpl-> X);
      mat Kinv = K.i();
      vec diff = pimpl-> y;
      mat dLLdK = -0.5 * Kinv + 0.5 * Kinv * diff * diff.t() * Kinv;
      for (size_t d = 0; d < grad.size(); d++) {
        mat dKdT = pimpl-> kernel-> derivate(d, pimpl-> X, pimpl-> X);
        grad[d] = trace(dLLdK * dKdT);
      }
      return ans;
    }

    double train(int max_iter, double tol) {
      nlopt::opt my_min(nlopt::LD_MMA, kernel-> n_params());
      my_min.set_max_objective(implementation::training_obj, this);
      my_min.set_xtol_rel(tol);
      my_min.set_maxeval(max_iter);

      my_min.set_lower_bounds(kernel-> get_lower_bounds());
      my_min.set_upper_bounds(kernel-> get_upper_bounds());

      double error; //final value of error function (myfunction)
      vector<double> x = kernel-> get_params();
      my_min.optimize(x, error);
      kernel-> set_params(x);
      return error;
    }
  };

  gp_reg::gp_reg() {
    pimpl = new implementation();
  }

  gp_reg::~gp_reg() {
    delete pimpl;
  }

  void gp_reg::set_kernel(const std::shared_ptr<kernel_class>& k) {
    pimpl-> kernel = k;
  }

  shared_ptr<kernel_class> gp_reg::get_kernel() const {
    return pimpl-> kernel;
  }

  void gp_reg::set_training_set(const arma::mat &X, const arma::vec& y) {
    pimpl-> X = X;
    pimpl-> y = y;
  }

  double gp_reg::train(const int max_iter, double tol) {
    return pimpl-> train(max_iter, tol);
  }

  mv_gauss gp_reg::full_predict(const arma::mat &new_data) const {
    return pimpl-> predict(new_data);
  }

  arma::vec gp_reg::predict(const arma::mat &new_data) const {
    mv_gauss g = pimpl-> predict(new_data);
    return g.get_mean();
  }
};

