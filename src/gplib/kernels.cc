#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {
  namespace kernels {
    struct squared_exponential::implementation {
      vector<double> params;
      vector<double> lower_bounds;
      vector<double> upper_bounds;

      double kernel(const vec &X, const vec &Y) {
        double sigma  = params[0];
        double lambda = params[1];
        mat diff = X - Y;
        mat tmp = (diff.t() * diff) / (-2.0 * (lambda * lambda));
        return sigma * sigma * exp(tmp(0,0));
      }

      mat eval(const arma::mat& X, const arma::mat& Y ) {
        mat ans(X.n_rows, Y.n_rows);
        for (size_t i = 0; i < X.n_rows; ++i) {
          for (size_t j = 0; j < Y.n_rows; ++j) {
            ans(i, j) = kernel(X.row(i).t(), Y.row(j).t());
          }
        }
        return ans + params[2] * params[2] * eye(X.n_rows, Y.n_rows);
      }

      double derivative_entry(size_t param_id, const vec &X, const vec &Y) {

        double sigma  = params[0];
        double lambda = params[1];
        mat diff = X - Y;

        if (param_id == 0) { // Sigma
          mat tmp = (diff.t() * diff) * -0.5 / (lambda * lambda);
          return 2.0 * sigma * exp(tmp(0, 0));
        }

        if (param_id == 1) { // lenght scale
          mat tmp = (kernel(X, Y) * (diff.t() * diff)) /
                    (lambda * lambda * lambda);
          return tmp(0, 0);
        }
        return 0;
      }

      mat derivate_wrt_inputs (size_t param_id, const arma::mat& X,
                               const arma::mat& Y){
        mat ans = zeros<mat> (X.n_rows, Y.n_rows);
        size_t row, col;
        bool u = X.size() < Y.size();
        if (u){
          row = param_id / X.n_rows;
          col = param_id % X.n_rows;
        } else{
          row = param_id / Y.n_rows;
          col = param_id % Y.n_rows;
        }

        for (size_t i = 0; i < X.n_rows; ++i) {
          for (size_t j = 0; j < Y.n_rows; ++j) {
            //Compute only the entries that are not 0
            if ((u && i == row) || (!u && j == row)){
              vec dXdT = zeros<vec> (X.n_cols);
              vec dYdT = zeros<vec> (Y.n_cols);
              if (u)
                dXdT(col) = 1;
              else
                dYdT(col) = 1;
              mat long_term = -0.5 * params[1] * (dXdT * X.row(i) - dXdT *
                              Y.row(j) - dYdT * X.row(i) + dYdT * Y.row(j) +
                              X.row(i) * dXdT - X.row(i) * dYdT - Y.row(j) *
                              dXdT + Y.row(j) * dYdT);
              cout << long_term.size();
              ans(i, j) = kernel(X.row(i).t(), Y.row(j).t()) * long_term(0, 0);
            }
          }
        }
        return ans;
      }

      mat derivative(size_t param_id, const arma::mat& X, const arma::mat& Y) {

        if (param_id < 2) {
          mat ans(X.n_rows, Y.n_rows);
          for (size_t i = 0; i < ans.n_rows; ++i) {
            for (size_t j = 0; j < ans.n_cols; ++j) {
              ans(i, j) = derivative_entry(param_id, X.row(i).t(),
                          Y.row(j).t());
            }
          }
          return ans;
        }
        if (param_id == 2)
          return 2.0 * params[2] * eye(X.n_rows, Y.n_rows);
        //Substract previous params
        mat ans = derivate_wrt_inputs (param_id - 2, X, Y);
        return ans;

      }
    }; // End of implementation.

    squared_exponential::squared_exponential() {
      pimpl = new implementation;
    }

    squared_exponential::squared_exponential(const vector<double> &params)
      : squared_exponential() {
      pimpl->params = params;
    }

    squared_exponential::~squared_exponential() {
      delete pimpl;
    }

    mat squared_exponential::eval(const arma::mat& X, const arma::mat& Y) const {
      return pimpl->eval(X, Y);
    }

    mat squared_exponential::derivate(size_t param_id, const arma::mat& X,
        const arma::mat& Y) const {

      return pimpl->derivative(param_id, X, Y);
    }

    size_t squared_exponential::n_params() const {
      return pimpl->params.size();
    }

    void squared_exponential::set_params(const vector<double> &params) {
      pimpl->params = params;
    }

    vector<double> squared_exponential::get_params() const {
      return pimpl->params;
    }

    void squared_exponential::set_lower_bounds(const vector<double> &lower_bounds) {
        pimpl-> lower_bounds = lower_bounds;
    }

    void squared_exponential::set_upper_bounds(const vector<double> &upper_bounds) {
        pimpl-> upper_bounds = upper_bounds;
    }
    vector<double> squared_exponential::get_lower_bounds() const {
        return pimpl->lower_bounds;
    }

    vector<double> squared_exponential::get_upper_bounds() const {
        return pimpl->upper_bounds;
    }
  };
};
