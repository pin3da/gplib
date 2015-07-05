#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib{
  namespace multioutput_kernels{
    struct lmc_kernel::implementation{
      vector<mat> params;
      vector<shared_ptr<kernel_class>> kernels;

      mat eval(const vector<mat> &X, unsigned int lf_number){
        //Comput cov mat total size;
        unsigned long total_rows = 0;
        for (unsigned int i = 0; i < X.size(); ++i){
          total_rows += X[i].n_rows;
        }

        //Compute cov mat
        mat cov(total_rows, total_rows);
        unsigned long first_row = 0, first_col = 0;
        for (unsigned int i = 0; i < X.size(); i++) {
          for (unsigned int j = 0; j < X.size(); j++) {
            mat cov_ab = zeros<mat> (X[i].n_rows, X[j].n_rows);
            for (unsigned int k = 0; k < lf_number; k++)
              cov_ab += params[k](i,j) * (kernels[k]->eval(X[i], X[j], i, j));
            cov.submat (first_row, first_col, first_row + X[i].n_rows - 1, first_col + X[j].n_rows - 1) = cov_ab;
            first_col += X[j].n_rows;
          }
          first_row += X[i].n_rows;
          first_col = 0;
        }

        return cov;

      }

    };

    lmc_kernel::lmc_kernel(){
      pimpl = new implementation;
    }

    lmc_kernel::lmc_kernel(const vector<shared_ptr<kernel_class>> &kernels, const vector<mat> &params) : lmc_kernel() {
      pimpl -> kernels = kernels;
      pimpl -> params = params;
    }

    lmc_kernel::~lmc_kernel(){
      delete pimpl;
    }

    mat lmc_kernel::eval(const vector<mat> &X, unsigned int lf_number){
      return pimpl -> eval(X, lf_number);
    }

    mat lmc_kernel::derivate(){
      return zeros<mat> (3, 3);
    }

    void lmc_kernel::set_params(const vector<mat> &params){
      pimpl -> params = params;
    }

    void lmc_kernel::set_kernels(const vector<shared_ptr<kernel_class>> &kernels){
      pimpl -> kernels = kernels;
    }

    vector<mat> lmc_kernel::get_params(){
      return pimpl -> params;
    }

    vector<shared_ptr<kernel_class>> lmc_kernel::get_kernels(){
      return pimpl -> kernels;
    }

  };
};
