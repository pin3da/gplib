#include "gplib.h"

using namespace arma;
using namespace std;

namespace gplib {
  namespace kernels {


    struct SquaredExponential::Implementation {
      vector<int> triangular_numbers = {0,1,3,6,10,15,21,28,36,45,55,66,78,91,105,120,
          136,153,171,190,210,231,253,276,300,325,351,378,
          406,435,465,496,528,561,595,630,666,703,741,780,
          820,861,903,946,990,1035,1081,1128,1176,1225,1275,
          1326,1378,1431};

      vector<double> params;
      mat lambda;

      double kernel(const vec &X, const vec &Y) {
        mat diff = X - Y;
        mat tmp = (diff.t() * lambda * diff);
        myassert(tmp.size() == 1);
        return params.back() * params.back() * exp(tmp(0, 0));
      }

      mat eval(const arma::mat& X, const arma::mat& Y, unsigned int idOut1, unsigned int idOut2) {
        mat ans(X.n_rows, Y.n_rows);
        for (size_t i = 0; i < X.n_cols; ++i) {
          for (size_t j = 0; j < Y.n_cols; ++j) {
            ans(i, j) = kernel(X.row(i).t(), Y.row(j).t());
          }
        }
        return ans;
      }

      double derivate_lambda(int a, int b, int i, int j) {

        if (a == i && b == i)
          return 2.0 * lambda(i, j);
        else if (a == i)
          return lambda(b, j);
        else if (b == i)
          return lambda(a, j);
        return 0;
      }

      double derivative_entry(unsigned int paramId, const vec &X, const vec &Y,
          unsigned int idOut1, unsigned int idOut2) {
        if (paramId == params.size() - 1) { // derivative with respect to sigma
          vec diff = X - Y;
          mat ans = diff.t() * lambda * diff;
          return 2.0 * params.back() * ans(0, 0);
        }
        else { // derivative with respect to one element of lambda
          // d K(X, Xp) / dLij =  K(X,Xp) * Tr [  (-1/2) (X- Xp) (X - Xp).T d(Lamda)/ dLij  ]
          int i = paramId / lambda.n_cols;
          int j = paramId % lambda.n_cols;

          mat diff = X - Y;
          mat d_lambda = lambda;
          for (size_t a = 0; a < lambda.n_rows; ++a)
            for (size_t b = 0; b < lambda.n_cols; ++b)
              d_lambda = derivate_lambda(a, b, i, j);

          mat tmp = mat(diff * diff.t())(0,0) * d_lambda;
          myassert(tmp.n_rows == tmp.n_cols);
          return kernel(X, Y) * trace(tmp);
        }
      }

      mat derivative(unsigned int paramId, const arma::mat& X, const arma::mat& Y,
          unsigned int idOut1, unsigned int idOut2) {

        mat ans(X.n_rows, Y.n_rows);
        for (size_t i = 0; i < ans.n_rows; ++i) {
          for (size_t j = 0; j < ans.n_cols; ++j) {
            ans(i, j) = derivative_entry(paramId, X.row(i).t(), Y.row(j).t(), idOut1, idOut2);
          }
        }
        return ans;
      }

      void setParams(const vector<double> &params) {
        int n = params.size() - 1;
        myassert(binary_search(triangular_numbers.begin(), triangular_numbers.end(), n));
        int index = lower_bound(triangular_numbers.begin(), triangular_numbers.end(), n) - triangular_numbers.begin();
        lambda = mat(index, index);
        int counter = 0;
        for (int i = 0; i < index; ++i) {
          for (int j = 0; j <= i; ++j) {
            lambda(i, j) = params[counter++];
          }
        }
      }
    };

    SquaredExponential::SquaredExponential() {

    }

    SquaredExponential::SquaredExponential(const vector<double> &params) {

    }

    SquaredExponential::~SquaredExponential() {

    }

    mat SquaredExponential::eval(const arma::mat& X, const arma::mat& Y, unsigned int idOut1, unsigned int idOut2) {
      return pimpl->eval(X, Y, idOut1, idOut2);
    }

    mat SquaredExponential::derivate(unsigned int paramId, const arma::mat& X, const arma::mat& Y,
        unsigned int idOut1, unsigned int idOut2) {

      return pimpl->derivative(paramId, X, Y, idOut1, idOut2);
    }

    unsigned int SquaredExponential::nparams() const {
      return pimpl->params.size();

    }

    void SquaredExponential::setParams(const vector<double> &params) {
      return pimpl->setParams(params);
    }

    vector<double> SquaredExponential::getParams() const {
      return pimpl->params;
    }

    vector<double> SquaredExponential::setLowerBounds() {

    }

    vector<double> SquaredExponential::getLowerBounds() const {

    }

    vector<double> SquaredExponential::setUpperBounds() {

    }

    vector<double> SquaredExponential::getUpperBounds() const {

    }
  };
};
