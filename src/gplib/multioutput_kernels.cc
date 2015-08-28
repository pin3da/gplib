#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib{
  namespace multioutput_kernels{
    struct lmc_kernel::implementation{
      vector<mat> params;
      vector<mat> A; // A * A.t() = params, where A is lower triangular.
      vector<shared_ptr<kernel_class>> kernels;

      mat eval(const vector<mat> &X) {
        //Comput cov mat total size;
        size_t lf_number = A.size();
        size_t total_rows = 0;
        for (size_t i = 0; i < X.size(); ++i) {
          total_rows += X[i].n_rows;
        }

        //Compute cov mat
        mat cov(total_rows, total_rows);
        size_t first_row = 0, first_col = 0;
        for (size_t i = 0; i < X.size(); i++) {
          for (size_t j = 0; j < X.size(); j++) {
            mat cov_ab = zeros<mat> (X[i].n_rows, X[j].n_rows);
            for (size_t k = 0; k < lf_number; k++)
              cov_ab += params[k](i, j) * (kernels[k]-> eval(X[i], X[j]));

            cov.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                first_col + X[j].n_rows - 1) = cov_ab;

            first_col += X[j].n_rows;
          }
          first_row += X[i].n_rows;
          first_col = 0;
        }
        return cov;
      }

      mat derivate(size_t param_id, const vector<mat> &X, const vector<mat> &Y,
        size_t id_out_1, size_t id_out_2) {

        size_t tot_rows = 0, tot_cols = 0;
        for (size_t i = 0; i < X.size(); ++i)
          tot_rows += X[i].n_rows;

        for (size_t i = 0; i < Y.size(); ++i)
          tot_cols += Y[i].n_rows;

        mat ans = zeros(tot_rows, tot_cols);
        for (size_t q = 0; q < params.size(); ++q) { // current latent fuction.
          if (param_id < params[q].size()) {
            size_t first_row = 0, first_col = 0;
            for (size_t i = 0; i < X.size(); i++) {
              for (size_t j = 0; j < Y.size(); j++) {
                mat ans_ab = zeros<mat> (X[i].n_rows, Y[j].n_rows);
                if (i == id_out_1 && j != id_out_1) {
                  ans_ab = A[q](j, id_out_2) * (kernels[q]-> eval(X[i], Y[j]));
                } else if (i != id_out_1 && j == id_out_1) {
                  ans_ab = A[q](i, id_out_2) * (kernels[q]-> eval(X[i], Y[j]));
                } else if (i == id_out_1 && j == id_out_1) {
                  ans_ab = 2.0 * A[q](id_out_1, id_out_2) *
                    (kernels[q]-> eval(X[i], X[j]));
                }

                ans.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                    first_col + X[j].n_rows - 1) = ans_ab;

                first_col += X[j].n_rows;
              }
              first_row += X[i].n_rows;
              first_col = 0;
            }
            return ans;
          }
          param_id -= params[q].size();
        }

        // from here they must be params of each kernel.
        for (size_t q = 0; q < kernels.size(); ++q) {
          if (param_id < kernels[q]-> n_params()) {
            size_t first_row = 0, first_col = 0;
            for (size_t i = 0; i < X.size(); i++) {
              for (size_t j = 0; j < Y.size(); j++) {
                mat ans_ab = params[q](i, j) * kernels[q]-> derivate(param_id, X[i], Y[j]);

                ans.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                    first_col + X[j].n_rows - 1) = ans_ab;

                first_col += X[j].n_rows;
              }
              first_row += X[i].n_rows;
              first_col = 0;
            }
            return ans;

           //  ans must be equals to kron(params[q], kernels[q]-> derivate(param_id, X, Y));
          }
          param_id -= kernels[q]-> n_params();
        }
        return ans;
      }

      void set_params(const vector<mat> &params) {
        A.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
          A[i] = chol(params[i]);
        }
      }
    };

    lmc_kernel::lmc_kernel() {
      pimpl = new implementation;
    }

    lmc_kernel::lmc_kernel(const vector<shared_ptr<kernel_class>> &kernels,
        const vector<mat> &params) : lmc_kernel() {
      pimpl-> kernels = kernels;
      pimpl-> params = params;
    }

    lmc_kernel::~lmc_kernel() {
      delete pimpl;
    }

    mat lmc_kernel::eval(const vector<mat> &X) const {
      return pimpl-> eval(X);
    }

    mat lmc_kernel::derivate(size_t param_id, const vector<mat> &X, const vector<mat> &Y,
        size_t id_out_1, size_t id_out_2) const {
      return pimpl-> derivate(param_id, X, Y, id_out_1, id_out_2);
    }

    void lmc_kernel::set_params(const vector<mat> &params) {
      // pimpl-> params = params;
      pimpl-> set_params(params);
    }

    void lmc_kernel::set_kernels(const vector<shared_ptr<kernel_class>> &kernels) {
      pimpl-> kernels = kernels;
    }

    vector<mat> lmc_kernel::get_params() const {
      return pimpl-> params;
    }

    vector<shared_ptr<kernel_class>> lmc_kernel::get_kernels() const {
      return pimpl-> kernels;
    }

  };
};
