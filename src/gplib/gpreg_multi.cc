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

    vec eval_mean(vector<mat> &data){
      unsigned long total_size;
      for (unsigned int i = 0; i < data.size(); i++){
        total_size += data[i].n_rows;
      }
      return zeros<vec> (total_size);
    }

    mv_gauss predict(const vector<mat> &new_data){
      //Add new data to observations
      vector<mat> M(X.size());
      unsigned long total_cols = 0;
      for (unsigned int i = 0; i < X.size(); i++){
        M[i] = join_vert (X[i], new_data[i]);
        total_cols += M[i].n_cols;
      }

      //Compute Covariance Matrix
      mat cov(total_cols, total_cols);
      unsigned long first_row = 0, first_col = 0;
      for (unsigned int i = 0; i < M.size(); i++){
        for (unsigned int j = 0; j < M.size(); j++){
          mat cov_ab = zeros<mat> (M[i].n_rows, M[j].n_cols);
          for (unsigned int k = 0; k < lf_number; k++)
            cov_ab += params(i,j) * (kernels[k]->eval(M[i], M[j], i, j));
          cov.submat (first_row, first_col, first_row + M[i].n_rows, first_col + M[j].n_cols) = cov_ab;
          first_col += M[j].n_cols;
        }
        first_row += M[i].n_rows;
      }

      //Set mean
      vec mean = eval_mean(M);
      //Set alredy observed Values
      vector<bool> observed(mean.n_rows, false);
      unsigned long start = 0;
      for (unsigned int i = 0; i < M.size(); i++){
        for (unsigned int j = 0; j < X[i].n_rows; j++)
          observed[start + j] = true;
        start += M[i].n_rows;
      }

      //Conditon Multivariate Gaussian
      mv_gauss gd(mean, cov);
      return gd.conditional(y, observed);
    }
  };
};
