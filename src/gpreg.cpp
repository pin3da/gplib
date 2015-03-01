
#include "gplib.h"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib {

  struct GPReg::Implementation {
    shared_ptr<Kernel> kernel;
    mat X; //Matrix of inputs
    vec y; //vector of outputs
    double noise;

    vec evalMean(const arma::mat& data) {
      // For the moment just use the zero mean
      return zeros<mat>(data.n_rows, data.n_rows);
    }

    MVGauss predict(const arma::mat& newData) {
      mat M = join_vert(X, newData);
      int N = X.n_rows, Nval = newData.n_rows;
      mat cov = kernel->eval(M,M) + noise*eye<mat>(N+Nval,N+Nval);
      vec mean = evalMean(M);
      MVGauss gd(mean, cov);
      vector<bool> observed(N+Nval, false);
      for (int i=0; i<N; i++) observed[i] = true;
      return gd.conditional(y, observed);
    }

    MVGauss marginal() {
      vec mean = evalMean(X);
      mat cov = kernel->eval(X,X) + noise*eye<mat>(X.n_rows,X.n_rows);
      return MVGauss(mean, cov);
    }

    double log_marginal() {
      return marginal().log_density(y);
    }

    void checkGrad(const vector<double>& deriv) {
      vector<double> params = kernel->getParams();
      int N = params.size();
      vector<double> nderiv(N);
      double eps = 1e-6;
      for (int i=0; i<N; i++) {
        params[i] += eps;
        kernel->setParams(params);
        double t1 = log_marginal();

        params[i] -= 2*eps;
        kernel->setParams(params);
        double t2 = log_marginal();
        nderiv[i] = (t1-t2)/(2*eps);
        params[i] += eps;
        kernel->setParams(params);
      }
      cout << "Numerical\tAnalitical\tDifference\n";
      for (int i=0; i<N; i++) {
        cout << nderiv[i] << "\t" << deriv[i] << "\t" <<
         nderiv[i] - deriv[i] << endl;
      }
    }

    static double trainingObj(const vector<double> &theta, vector<double> &grad, void *fdata) {
      Implementation *pimpl = (Implementation*) fdata;
      pimpl->kernel->setParams(theta);
      double ans = pimpl->log_marginal();

      vec mx = pimpl->evalMean(pimpl->X);
      mat K = pimpl->kernel->eval(pimpl->X,pimpl->X);
      mat Kinv = K.i();
      vec diff = pimpl->y;
      mat dLLdK = -0.5*Kinv + 0.5*Kinv*diff*diff.t()*Kinv;
      for (unsigned int d=0; d<grad.size(); d++) {
        mat dKdT = pimpl->kernel->derivate(d, pimpl->X, pimpl->X);
        grad[d] = trace(dLLdK * dKdT);
      }
      pimpl->checkGrad(grad);

      return ans;
    }

    void train(int maxIter) {
      using namespace nlopt;
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
      kernel->setParams(x);
    }
  };

  GPReg::GPReg() {
    pimpl = new Implementation();
  }

  GPReg::~GPReg() {
    delete pimpl;
  }

  void GPReg::setKernel(const std::shared_ptr<Kernel>& k) {
    pimpl->kernel = k;
  }

  shared_ptr<Kernel> GPReg::getKernel() const {
    return pimpl->kernel;
  }

  void GPReg::setTrainingSet(const arma::mat &X, const arma::vec& y) {
    pimpl->X = X;
    pimpl->y = y;
  }

  //void train();

  MVGauss GPReg::fullPredict(const arma::mat& newData) const {
    return pimpl->predict(newData);
  }

  arma::vec GPReg::predict(const arma::mat& newData) const {
    MVGauss g = pimpl->predict(newData);
    return g.getMean();
  }

};
