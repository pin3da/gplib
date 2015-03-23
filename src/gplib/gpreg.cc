
#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib {

  struct gp_reg::implementation {
    shared_ptr<kernel_class> kernel;
    mat X; //Matrix of inputs
    vec y; //vector of outputs
    double noise;

    vec eval_mean(const arma::mat& data) {
      // For the moment just use the zero mean
      return zeros<mat>(data.n_rows, data.n_rows);
    }

    void set_noise(double noise){
      this->noise = noise;
    }

    double get_noise(){
      return noise;
    }

    mv_gauss predict(const arma::mat& new_data) {
      mat M = join_vert(X, new_data);
      int n = X.n_rows, n_val = new_data.n_rows;
      mat cov = kernel->eval(M, M, 0, 0) + noise * eye<mat>(n + n_val, n + n_val);
      vec mean = eval_mean(M);
      mv_gauss gd(mean, cov);
      vector<bool> observed(n + n_val, false);
      for (int i = 0; i < n; ++i)
        observed[i] = true;
      return gd.conditional(y, observed);
    }

    mv_gauss marginal() {
      vec mean = eval_mean(X);
      mat cov = kernel->eval(X, X, 0, 0) + noise * eye<mat>(X.n_rows, X.n_rows);
      return mv_gauss(mean, cov);
    }

    double log_marginal() {
      return marginal().log_density(y);
    }

    void check_grad(const vector<double>& deriv) {
      vector<double> params = kernel->get_params();
      int n = params.size();
      vector<double> nderiv(n);
      double eps = 1e-6;
      for (int i = 0; i < n; i++) {
        params[i] += eps;
        kernel->set_params(params);
        double t1 = log_marginal();

        params[i] -= 2*eps;
        kernel->set_params(params);
        double t2 = log_marginal();
        nderiv[i] = (t1 - t2) / (2 * eps);
        params[i] += eps;
        kernel->set_params(params);
      }
      cout << "Numerical\tAnalitical\tDifference\n";
      for (int i = 0; i < n; i++) {
        cout << nderiv[i] << "\t" << deriv[i] << "\t" <<
         nderiv[i] - deriv[i] << endl;
      }
    }

    static double training_obj(const vector<double> &theta, vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;
      pimpl->kernel->set_params(theta);
      double ans = pimpl->log_marginal();

      vec mx = pimpl->eval_mean(pimpl->X);
      mat K = pimpl->kernel->eval(pimpl->X, pimpl->X, 0, 0);
      mat Kinv = K.i();
      vec diff = pimpl->y;
      mat dLLdK = -0.5 * Kinv + 0.5 * Kinv * diff * diff.t() * Kinv;
      for (size_t d = 0; d < grad.size(); d++) {
        mat dKdT = pimpl->kernel->derivate(d, pimpl->X, pimpl->X, 0, 0);
        grad[d] = trace(dLLdK * dKdT);
      }
      pimpl->check_grad(grad);

      return ans;
    }

    void train(int max_iter) {
      nlopt::opt my_min(nlopt::LD_MMA,kernel->n_params());
      my_min.set_min_objective(implementation::training_obj, this);
      my_min.set_xtol_rel(1e-4);
      my_min.set_maxeval(max_iter);
      my_min.set_lower_bounds(kernel->get_lower_bounds());
      my_min.set_upper_bounds(kernel->get_upper_bounds());

      double error; //final value of error function (myfunction)
      vector<double> x = kernel->get_params();
      //result r = my_min.optimize(x,error);
      my_min.optimize(x, error);
      kernel->set_params(x);
    }
  };

  gp_reg::gp_reg() {
    pimpl = new implementation();
  }

  gp_reg::~gp_reg() {
    delete pimpl;
  }

  void gp_reg::set_kernel(const std::shared_ptr<kernel_class>& k) {
    pimpl->kernel = k;
  }

  shared_ptr<kernel_class> gp_reg::get_kernel() const {
    return pimpl->kernel;
  }

  void gp_reg::set_training_set(const arma::mat &X, const arma::vec& y) {
    pimpl->X = X;
    pimpl->y = y;
  }

  void gp_reg::train(const int max_iter){
    pimpl->train(max_iter);
  }

  mv_gauss gp_reg::full_predict(const arma::mat& new_data) const {
    return pimpl->predict(new_data);
  }

  arma::vec gp_reg::predict(const arma::mat& new_data) const {
    mv_gauss g = pimpl->predict(new_data);
    return g.get_mean();
  }

};
