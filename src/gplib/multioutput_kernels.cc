#include "gplib.hpp"
#include <algorithm>

using namespace arma;
using namespace std;

namespace gplib{
  namespace multioutput_kernels{
    struct lmc_kernel::implementation{
      vector<mat> B;
      vector<mat> A; // A * A.t() = B, where A is lower triangular.
      vector<shared_ptr<kernel_class>> kernels;
      vector<double> lower_bounds;
      vector<double> upper_bounds;

      mat eval(const vector<mat> &X, const vector<mat> &Y) {
        //Comput cov mat total size;
        size_t total_rows = 0, total_cols = 0;
        for (size_t i = 0; i < X.size(); ++i) {
          total_rows += X[i].n_rows;
        }
        for (size_t i = 0; i < Y.size(); ++i) {
          total_cols += Y[i].n_rows;
        }

        //Compute cov mat
        mat cov(total_rows, total_cols);
        size_t first_row = 0, first_col = 0;
        for (size_t i = 0; i < X.size(); i++) {
          for (size_t j = 0; j < Y.size(); j++) {
            mat cov_ab = zeros<mat> (X[i].n_rows, Y[j].n_rows);
            for (size_t k = 0; k < B.size(); k++) {
              cov_ab += B[k](i, j) * (kernels[k]-> eval(X[i], Y[j]));
            }

            cov.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                first_col + Y[j].n_rows - 1) = cov_ab;

            first_col += Y[j].n_rows;
          }
          first_row += X[i].n_rows;
          first_col = 0;
        }
        return cov;
      }


      mat derivate(size_t param_id, const vector<mat> &X, const vector<mat> &Y) {

        size_t tot_rows = 0, tot_cols = 0;
        for (size_t i = 0; i < X.size(); ++i)
          tot_rows += X[i].n_rows;

        for (size_t i = 0; i < Y.size(); ++i)
          tot_cols += Y[i].n_rows;

        mat ans = zeros(tot_rows, tot_cols);
        for (size_t q = 0; q < B.size(); ++q) { // current latent fuction.
          if (param_id < B[q].size()) {
            size_t id_out_1 = param_id / B[q].n_rows;
            size_t id_out_2 = param_id % B[q].n_rows;
            size_t first_row = 0, first_col = 0;
            for (size_t i = 0; i < X.size(); i++) {
              for (size_t j = 0; j < Y.size(); j++) {
                mat ans_ab = zeros<mat> (X[i].n_rows, Y[j].n_rows);
                if (i == id_out_1 && j == id_out_1) {
                  ans_ab = A[q](id_out_1, id_out_2) * (kernels[q]-> eval(X[i], X[j]));
                }
                else if (j == id_out_1) {
                  ans_ab = A[q](i, id_out_2) * (kernels[q]-> eval(X[i], Y[j]));
                } else if (i == id_out_1) {
                  ans_ab = A[q](j, id_out_2) * (kernels[q]-> eval(X[i], Y[j]));
                }
                if (i * X.size() + j == param_id)
                  ans.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                      first_col + X[j].n_rows - 1) = ans_ab;
                first_col += X[j].n_rows;
              }
              first_row += X[i].n_rows;
              first_col = 0;
            }
            return ans;
          }
          param_id -= B[q].size();
        }

        // from here they must be params of each kernel.
        for (size_t q = 0; q < kernels.size(); ++q) {
          if (param_id < kernels[q]-> n_params()) {
            size_t first_row = 0, first_col = 0;
            for (size_t i = 0; i < X.size(); i++) {
              for (size_t j = 0; j < Y.size(); j++) {
                mat ans_ab = B[q](i, j) * kernels[q]-> derivate(param_id, X[i], Y[j]);

                ans.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                    first_col + X[j].n_rows - 1) = ans_ab;

                first_col += X[j].n_rows;
              }
              first_row += X[i].n_rows;
              first_col = 0;
            }
            return ans;

            //  ans must be equals to kron(B[q], kernels[q]-> derivate(param_id, X, Y));
          }
          param_id -= kernels[q]-> n_params();
        }
        return ans;
      }

      vector<double> get_params() {
        //set total size of vector
        if (B.size() <= 0)
          return vector<double> ({-1.0});
        size_t t_size = B.size() + B[0].size();
        for (size_t k = 0; k< kernels.size(); ++k)
          t_size += kernels[k]->  n_params();
        vector<double> ans(t_size);
        size_t iter = 0;
        for (size_t q = 0; q < B.size(); ++q) {
          copy (B[q].begin(), B[q].end(), ans.begin() + iter);
          iter += B[q].size();
        }
        for (size_t k = 0; k < kernels.size(); ++k) {
          copy (kernels[k]-> get_params().begin(), kernels[k]-> get_params().end(), ans.begin() + iter);
          iter += kernels[k]-> n_params();
        }
        return ans;
      }

      void set_params(const vector<double> &params, size_t n_outputs = -1) {
        if (n_outputs == -1){
          if (B.size() > 0)
            n_outputs = B[0].n_cols;
          else
            throw logic_error("Parameters Uninitialized");//Throw exception here
        }
        size_t t_size = B.size() * n_outputs * n_outputs;
        for (size_t k = 0; k < kernels.size(); ++k)
          t_size += kernels[k]-> n_params();
        if(t_size != params.size()){
          throw length_error("Wrong parameter vector size");
        }
        size_t iter =0;
        for (size_t q = 0; q < B.size(); ++q) {
          if (B[q].size() <= 0)
            B[q] = mat(n_outputs, n_outputs, fill::zeros);
          for (size_t i = 0; i < B[0].n_rows; ++i) {
            for (size_t j = 0; j < B[0].n_cols; ++j) {
              B[q](i, j) = params[iter];
              ++iter;
            }
          }
        }
        for (size_t k = 0; k < kernels.size(); ++k) {
          vector<double> subparams(params.begin() + iter, params.begin() + iter + kernels[k]-> n_params());
          kernels[k]->set_params(subparams);
          iter += (kernels[k]-> n_params() + 1);
        }
      }

      void set_params_k(const vector<mat> &params) {
        A.resize(params.size());
        B.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
          A[i] = chol(params[i]);
          B[i] = params[i];
        }
      }

      void set_param(size_t q, size_t a, size_t b, double param) {
        if (q > B.size())
          throw out_of_range("latent function id out of range");
        if (a > B[q].n_rows || b > B[q].n_cols)
          throw out_of_range("Param id out of range");
        B[q](a, b) = param;
        A[q] = chol(B[q]);
      }

      void set_param(size_t q, size_t param_id, double param) {
        if (q > kernels.size())
          throw out_of_range("Kernel id out of range");
        vector<double> params = kernels[q]-> get_params();
        if (param_id > params.size())
          throw out_of_range("Param id out of range");
        params[param_id] = param;
        kernels[q]-> set_params(params);
      }

      double get_param(size_t q, size_t a, size_t b) {
        if (q > B.size())
          throw out_of_range("latent function id out of range");
        if (a > B[q].n_rows || b > B[q].n_cols)
          throw out_of_range("Param id out of range");
        return B[q](a, b);
      }

      double get_param(size_t q, size_t param_id) {
        if (q > kernels.size())
          throw out_of_range("Kernel id out of range");
        const vector<double> &params = kernels[q]-> get_params();
        if (param_id > params.size())
          throw out_of_range("Param id out of range");
        return params[param_id];
      }

      size_t n_params() {
        size_t ans = 0;
        for (size_t i = 0; B.size(); ++i)
          ans += B[i].size();

        for (size_t i = 0; i < kernels.size(); ++i)
          ans += kernels[i]->n_params();

        return ans;
      }
    };

    lmc_kernel::lmc_kernel() {
      pimpl = new implementation;
    }

    lmc_kernel::lmc_kernel(const vector<shared_ptr<kernel_class>> &kernels,
        const vector<mat> &params) : lmc_kernel() {
      pimpl-> kernels = kernels;
      pimpl-> set_params_k(params);
    }

    lmc_kernel::~lmc_kernel() {
      delete pimpl;
    }

    mat lmc_kernel::eval(const vector<mat> &X, const vector<mat> &Y) const {
      return pimpl-> eval(X, Y);
    }

    mat lmc_kernel::derivate(size_t param_id, const vector<mat> &X, const vector<mat> &Y) const {
      return pimpl-> derivate(param_id, X, Y);
    }

    size_t lmc_kernel::n_params() const {
      return pimpl-> n_params();
    }

    void lmc_kernel::set_params_k(const vector<mat> &params) {
      pimpl-> set_params_k(params);
    }

    void lmc_kernel::set_params(const vector<double> &params, size_t n_outputs) {
      // TODO:
      pimpl-> set_params(params, n_outputs);
    }

    void lmc_kernel::set_param(size_t q, size_t a, size_t b, double param) {
      pimpl-> set_param(q, a, b, param);
    }

    void lmc_kernel::set_param(size_t q, size_t param_id, double param) {
      pimpl-> set_param(q, param_id, param);
    }

    void lmc_kernel::set_kernels(const vector<shared_ptr<kernel_class>> &kernels) {
      pimpl-> kernels = kernels;
      pimpl-> B.resize (kernels.size());
      pimpl-> A.resize (kernels.size());
    }

    double lmc_kernel::get_param(size_t q, size_t a, size_t b) const {
      return pimpl-> get_param(q, a, b);
    }

    double lmc_kernel::get_param(size_t q, size_t param_id) const {
      return pimpl-> get_param(q, param_id);
    }

    vector<double> lmc_kernel::get_params() const {
      // TODO:
      return pimpl->get_params();
    }

    vector<mat> lmc_kernel::get_params_k() const {
      return pimpl-> B;
    }

    vector<shared_ptr<kernel_class>> lmc_kernel::get_kernels() const {
      return pimpl-> kernels;
    }

    void lmc_kernel::set_lower_bounds(const double &lower_bounds) {
      vector<double> lower_bound(pimpl-> n_params(), lower_bounds);
      set_lower_bounds(lower_bound);
    }

    void lmc_kernel::set_upper_bounds(const double &upper_bounds) {
      vector<double> upper_bound(pimpl-> n_params(), upper_bounds);
      set_lower_bounds(upper_bound);
    }

    void lmc_kernel::set_lower_bounds(const vector<double> &lower_bounds) {
      pimpl-> lower_bounds = lower_bounds;
    }

    void lmc_kernel::set_upper_bounds(const vector<double> &upper_bounds) {
      pimpl-> upper_bounds = upper_bounds;
    }

    vector<double> lmc_kernel::get_lower_bounds() const {
      return pimpl-> lower_bounds;
    }

    vector<double> lmc_kernel::get_upper_bounds() const {
      return pimpl-> upper_bounds;
    }

  };
};
