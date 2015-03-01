
#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib {

  struct gp_reg::implementation {
    shared_ptr<kernel> kernel;
    mat X; //Matrix of inputs
    vec y; //vector of outputs
    double noise;

    vec eval_mean(const arma::mat& data) {
      // For the moment just use the zero mean
      return zeros<mat>(data.n_rows, data.n_rows);
    }

    mv_gauss predict(const arma::mat& newData) {
      mat M = join_vert(X, newData);
      int N = X.n_rows, Nval = newData.n_rows;
      mat cov = kernel->eval(M,M) + noise*eye<mat>(N+Nval,N+Nval);
      vec mean = eval_mean(M);
      mv_gauss gd(mean, cov);
      vector<bool> observed(N+Nval, false);
      for (int i=0; i<N; i++) observed[i] = true;
      return gd.conditional(y, observed);
    }

    mv_gauss marginal() {
      vec mean = eval_mean(X);
      mat cov = kernel->eval(X,X) + noise*eye<mat>(X.n_rows,X.n_rows);
      return mv_gauss(mean, cov);
    }

    double log_marginal() {
      return marginal().log_density(y);
    }

    void check_grad(const vector<double>& deriv) {
      vector<double> params = kernel->get_params();
      int N = params.size();
      vector<double> nderiv(N);
      double eps = 1e-6;
      for (int i=0; i<N; i++) {
        params[i] += eps;
        kernel->set_params(params);
        double t1 = log_marginal();

        params[i] -= 2*eps;
        kernel->set_params(params);
        double t2 = log_marginal();
        nderiv[i] = (t1-t2)/(2*eps);
        params[i] += eps;
        kernel->set_params(params);
      }
      cout << "Numerical\tAnalitical\tDifference\n";
      for (int i=0; i<N; i++) {
        cout << nderiv[i] << "\t" << deriv[i] << "\t" <<
         nderiv[i] - deriv[i] << endl;
      }
    }

    static double training_obj(const vector<double> &theta, vector<double> &grad, void *fdata) {
      implementation *pimpl = (implementation*) fdata;
      pimpl->kernel->set_params(theta);
      double ans = pimpl->log_marginal();

      vec mx = pimpl->eval_mean(pimpl->X);
      mat K = pimpl->kernel->eval(pimpl->X,pimpl->X);
      mat Kinv = K.i();
      vec diff = pimpl->y;
      mat dLLdK = -0.5*Kinv + 0.5*Kinv*diff*diff.t()*Kinv;
      for (size_t d=0; d<grad.size(); d++) {
        mat dKdT = pimpl->kernel->derivate(d, pimpl->X, pimpl->X);
        grad[d] = trace(dLLdK * dKdT);
      }
      pimpl->check_grad(grad);

      return ans;
    }

    void train(int maxIter) {
      nlopt::opt mymin(LD_MMA,kernel->nparams());
      mymin.set_min_objective(Implementation::trainingObj, this);
      mymin.set_xtol_rel(1e-4);
      mymin.set_maxeval(maxIter);
      mymin.set_lower_bounds(kernel->getLowerBounds());
      mymin.set_upper_bounds(kernel->getUpperBounds());

      double error; //final value of error function (myfunction)
      vector<double> x = kernel->getParams();
      //result r = mymin.optimize(x,error);
      mymin.optimize(x,error);
      kernel->set_params(x);
    }
  };

  gp_reg::gp_reg() {
    pimpl = new implementation();
  }

  gp_reg::~gp_reg() {
    delete pimpl;
  }

  void gp_reg::set_kernel(const std::shared_ptr<Kernel>& k) {
    pimpl->kernel = k;
  }

  shared_ptr<Kernel> gp_reg::get_kernel() const {
    return pimpl->kernel;
  }

  void gp_reg::set_training_set(const arma::mat &X, const arma::vec& y) {
    pimpl->X = X;
    pimpl->y = y;
  }

  //void train();

  mv_gauss gp_reg::full_predict(const arma::mat& newData) const {
    return pimpl->predict(newData);
  }

  arma::vec gp_reg::predict(const arma::mat& newData) const {
    mv_gauss g = pimpl->predict(newData);
    return g.get_mean();
  }

};
