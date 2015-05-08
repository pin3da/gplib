#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib{

  struct gp_reg_multi::implementation{
    size_t lf_number;
    vector<shared_ptr<kernel_class>> kernels;
    vector<mat> X;
    vec y;
    mat params;

    vec eval_mean(vector<mat> &data) {
      unsigned long total_size = 0;
      for (unsigned int i = 0; i < data.size(); i++) {
        total_size += data[i].n_rows;
      }
      return zeros<vec> (total_size);
    }

    mv_gauss predict(const vector<mat> &new_data) {
      //Add new data to observations
      vector<mat> M(X.size());
      unsigned long total_rows = 0;
      for (unsigned int i = 0; i < X.size(); i++) {
        M[i] = join_vert (X[i], new_data[i]);
        total_rows += M[i].n_rows;
      }

      //Compute Covariance Matrixkernels[k]->eval(M[i], M[j], i, j))
      mat cov(total_rows, total_rows);
      unsigned long first_row = 0, first_col = 0;
      for (unsigned int i = 0; i < M.size(); i++) {
        for (unsigned int j = 0; j < M.size(); j++) {
          mat cov_ab = zeros<mat> (M[i].n_rows, M[j].n_rows);
          for (unsigned int k = 0; k < lf_number; k++)
            cov_ab += params(i,j) * (kernels[k]->eval(M[i], M[j], i, j));
          cov.submat (first_row, first_col, first_row + M[i].n_rows - 1, first_col + M[j].n_rows - 1) = cov_ab;
          first_col += M[j].n_rows;
        }
        first_row += M[i].n_rows;
        first_col = 0;
      }
      //Set mean
      vec mean = eval_mean(M);
      //Set alredy observed Values
      vector<bool> observed(mean.n_rows, false);
      unsigned long start = 0;
      for (unsigned int i = 0; i < M.size(); i++) {
        for (unsigned int j = 0; j < X[i].n_rows; j++)
          observed[start + j] = true;
        start += M[i].n_rows;
      }
      //Conditon Multivariate Gaussian
      mv_gauss gd(mean, cov);
      return gd.conditional(y, observed);
    }

  };

  gp_reg_multi::gp_reg_multi(){
    pimpl = new implementation();
  }

  gp_reg_multi::~gp_reg_multi(){
    delete pimpl;
  }

  void gp_reg_multi::set_kernels(const vector<shared_ptr<kernel_class>> &k){
    pimpl->kernels = k;
    pimpl->lf_number = k.size();
  }

  void gp_reg_multi::set_training_set(const vector<mat> &X, const vec & y){
    pimpl->X = X;
    pimpl->y = y;
  }
  //Temporal!!!!
  void gp_reg_multi::set_params(const mat &params){
    pimpl->params = params;
  }

  void gp_reg_multi::set_lf_number(const int lf_number){
    pimpl->lf_number = lf_number;
  }
  mv_gauss gp_reg_multi::full_predict(const vector<mat> &new_data){
    return pimpl->predict(new_data);
  }
};
