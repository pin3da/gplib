#include "gplib.hpp"
#include <nlopt.hpp>

using namespace arma;
using namespace std;

namespace gplib{

  struct gp_reg_multi::implementation{
    size_t lf_number;
    shared_ptr<multioutput_kernel_class> kernel;
    vector<mat> X;
    vector<vec> y;

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
      vec fill_y;
      unsigned long total_rows = 0;
      for (unsigned int i = 0; i < X.size(); i++) {
        M[i] = join_vert (X[i], new_data[i]);
        fill_y = join_cols<mat> (fill_y, y[i]);
        fill_y = join_cols<mat> (fill_y, zeros<vec>(new_data[i].n_rows));
        total_rows += M[i].n_rows;
      }

      //Compute Covariance
      mat cov = kernel -> eval(M, M);
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
      return gd.conditional(fill_y, observed);
    }

  };

  gp_reg_multi::gp_reg_multi(){
    pimpl = new implementation();
  }

  gp_reg_multi::~gp_reg_multi(){
    delete pimpl;
  }

  void gp_reg_multi::set_kernel(const shared_ptr<multioutput_kernel_class> &k){
    pimpl->kernel = k;
    pimpl->lf_number = k->get_kernels().size();
  }

  void gp_reg_multi::set_training_set(const vector<mat> &X, const vector<vec> & y){
    pimpl->X = X;
    pimpl->y = y;
  }
  //Temporal!!!!

  void gp_reg_multi::set_lf_number(const int lf_number){
    pimpl->lf_number = lf_number;
  }
  mv_gauss gp_reg_multi::full_predict(const vector<mat> &new_data){
    return pimpl->predict(new_data);
  }
};
