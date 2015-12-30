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

      void default_constructor(size_t lf_number, size_t n_outputs) {
        vector<double> kernel_params(3, 0.1);
        kernels.clear();
        for (size_t i = 0; i < lf_number; ++i)
          kernels.push_back(make_shared<kernels::squared_exponential>\
              (kernel_params));


        A = vector<mat>(kernels.size(), eye<mat>(n_outputs, n_outputs));
        B = vector<mat>(kernels.size(), eye<mat>(n_outputs, n_outputs));
      }

      mat eval(const vector<mat> &X, const vector<mat> &Y) {
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

      mat derivate_wrt_data_an(size_t param_id, const vector<mat> &X, const
        vector<mat> &Y, size_t ans_rows, size_t ans_cols,
        bool diag = false) {

        //Put together a vector of sizes
        vector<size_t> x_sizes (X.size());
        vector<size_t> y_sizes (Y.size());
        for (size_t i = 0; i < X.size(); ++i)
          x_sizes[i] = X[i].size();
        for (size_t i = 0; i < Y.size(); ++i)
          y_sizes[i] = Y[i].size();


        //Set which size should be used to locate the pseudo-input
        vector<size_t> &U = (X[0].size() < Y[0].size()) ? x_sizes : y_sizes;
        bool u = X[0].size() < Y[0].size();

        //Find which matrix contains the pseudo-input of those in the vector
        size_t which_u;
        for (which_u = 0; which_u < U.size(); ++which_u) {
          if (param_id < U[which_u])
            break;
          param_id -= U[which_u];
        }

        //Compute cov mat
        mat cov = zeros<mat> (ans_rows, ans_cols);
        size_t first_row = 0, first_col = 0;
        for (size_t i = 0; i < X.size(); i++) {
          for (size_t j = 0; j < Y.size(); j++) {
            if((diag && i == j) || !diag){
              //Only compute the entries which aren't 0
              if ((u && i == which_u) || (!u && j == which_u)) {
                mat cov_ab = zeros<mat> (X[i].n_rows, Y[j].n_rows);
                for (size_t k = 0; k < B.size(); k++) {
                  cov_ab += B[k](i, j) * (kernels[k]-> derivate(param_id +
                            kernels[k]-> n_params(), X[i], Y[j], diag));
                }
                cov.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                  first_col + Y[j].n_rows - 1) = cov_ab;
              }
            }
            first_col += Y[j].n_rows;
          }
          first_row += X[i].n_rows;
          first_col = 0;
        }
        return cov;
      }

      mat derivate_wrt_data_num(size_t param_id, const vector<mat> &X,
        const vector<mat> &Y, size_t ans_rows, size_t ans_cols,
        bool diag = false){

        //Puut together a vector of sizes
        vector<size_t> x_sizes (X.size());
        vector<size_t> y_sizes (Y.size());
        for (size_t i = 0; i < X.size(); ++i)
          x_sizes[i] = X[i].size();
        for (size_t i = 0; i < Y.size(); ++i)
          y_sizes[i] = Y[i].size();

        //Set which size should be used to locate the pseudo-input
        vector<size_t> &U = (X[0].size() < Y[0].size()) ? x_sizes : y_sizes;
        bool u = X[0].size() < Y[0].size();

        //Find which matrix contains the pseudo-input of those in the vector
        size_t which_u;
        for (which_u = 0; which_u < U.size(); ++which_u) {
          if (param_id < U[which_u])
            break;
          param_id -= U[which_u];
        }

        size_t row, col;
        mat ans = zeros<mat>(ans_rows, ans_cols);
        vector<mat> tmp;
        double eps = 1e-5;
        if (u){
          tmp = X;
          row = param_id / tmp[which_u].n_cols;
          col = param_id % tmp[which_u].n_cols;
          tmp[which_u](row, col) += eps;
          ans = eval(tmp, Y);
          tmp[which_u](row, col) -= 2 * eps;
          ans -= eval(tmp, Y);
          ans = ans / (2 * eps);
        } else{
          tmp = Y;
          row = param_id / tmp[which_u].n_cols;
          col = param_id % tmp[which_u].n_cols;
          tmp[which_u](row, col) += eps;
          ans = eval(X, tmp);
          tmp[which_u](row, col) -= 2 * eps;
          ans -= eval(X, tmp);
          ans = ans / (2 * eps);
        }
        if (X[0].size() == Y[0].size()) ans += ans.t();
        if (diag) return diagmat(ans);
        return ans;

      }

      mat derivative_wrt_B(size_t q, size_t param_id, const vector<mat> &X,
        const vector<mat> &Y, size_t ans_rows, size_t ans_cols,
        bool diag = false){

        mat ans = zeros<mat>(ans_rows, ans_cols);
        size_t id_out_1 = param_id / B[q].n_rows;
        size_t id_out_2 = param_id % B[q].n_rows;
        size_t first_row = 0, first_col = 0;
        for (size_t i = 0; i < X.size(); i++) {
          for (size_t j = 0; j < Y.size(); j++) {
            if((diag && i == j) || !diag){
              mat ans_ab = zeros<mat> (X[i].n_rows, Y[j].n_rows);
              if (i == id_out_1 && j == id_out_1) {
                ans_ab = A[q](id_out_1, id_out_2) *
                         (kernels[q]-> eval(X[i], Y[j]));
              }
              else if (j == id_out_1) {
                ans_ab = A[q](i, id_out_2) * (kernels[q]-> eval(X[i], Y[j]));
              } else if (i == id_out_1) {
                ans_ab = A[q](j, id_out_2) * (kernels[q]-> eval(X[i], Y[j]));
              }
              if (i * X.size() + j == param_id){
                if (diag)
                  ans_ab = diagmat(ans_ab);
                ans.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                    first_col + Y[j].n_rows - 1) = ans_ab;
              }
            }
            first_col += Y[j].n_rows;
          }
          first_row += X[i].n_rows;
          first_col = 0;
        }
        return ans;
      }

      mat derivative_wrt_kernels(size_t q, size_t param_id, const vector<mat>
        &X, const vector<mat> &Y, size_t ans_rows, size_t ans_cols,
        bool diag = false){
        mat ans = zeros<mat>(ans_rows, ans_cols);
        size_t first_row = 0, first_col = 0;
        for (size_t i = 0; i < X.size(); i++) {
          for (size_t j = 0; j < Y.size(); j++) {
            if ((diag && i == j) || !diag){
              mat ans_ab = B[q](i, j) *
                           kernels[q]-> derivate(param_id, X[i], Y[j], diag);
              ans.submat (first_row, first_col, first_row + X[i].n_rows - 1,
                  first_col + Y[j].n_rows - 1) = ans_ab;
            }
            first_col += Y[j].n_rows;
          }
          first_row += X[i].n_rows;
          first_col = 0;
        }
        return ans;
      }

      mat derivate(size_t param_id, const vector<mat> &X, const vector<mat> &Y,
        bool diag = false) {

        size_t tot_rows = 0, tot_cols = 0;
        for (size_t i = 0; i < X.size(); ++i)
          tot_rows += X[i].n_rows;

        for (size_t i = 0; i < Y.size(); ++i)
          tot_cols += Y[i].n_rows;

        for (size_t q = 0; q < B.size(); ++q) { // current latent fuction.
          if (param_id < B[q].size())
            return derivative_wrt_B(q, param_id, X, Y, tot_rows, tot_cols,
            diag);
          param_id -= B[q].size();
        }

        // from here they must be params of each little kernel.
        for (size_t q = 0; q < kernels.size(); ++q) {
          if (param_id < kernels[q]-> n_params())
            return derivative_wrt_kernels(q, param_id, X, Y, tot_rows, tot_cols,
            diag);
            //  ans must be equal to
            //  kron(B[q], kernels[q]-> derivate(param_id, X, Y));
          param_id -= kernels[q]-> n_params();
        }

        return derivate_wrt_data_an(param_id, X, Y, tot_rows, tot_cols, diag);
      }


      vector<double> get_params() {
        //set total size of vector
        if (A.size() <= 0 || A[0].size() <= 0)
          return vector<double> ();

        size_t t_size = A.size() * A[0].size();
        for (size_t k = 0; k < kernels.size(); ++k)
          t_size += kernels[k]->  n_params();

        vector<double> ans(t_size);
        size_t iter = 0;
        for (size_t q = 0; q < A.size(); ++q) {
          for (size_t i = 0; i < A[q].n_rows; ++i) {
            for (size_t j = 0; j < A[q].n_cols; ++j) {
              ans[iter] = A[q](i, j);
              iter++;
            }
          }
        }

        vector<double> tmp;
        for (size_t k = 0; k < kernels.size(); ++k) {
          tmp = kernels[k]-> get_params();
          copy (tmp.begin(), tmp.end(), ans.begin() + iter);
          iter += kernels[k]-> n_params();
        }
        return ans;
      }

      void set_params(const vector<double> &params, int n_outputs = -1) {
        if (n_outputs == -1){
          if (A.size() > 0 && A[0].size() > 0)
            n_outputs = A[0].n_cols;
          else
            throw logic_error("Parameters Uninitialized");
        }
        size_t t_size = A.size() * n_outputs * n_outputs;
        for (size_t k = 0; k < kernels.size(); ++k)
          t_size += kernels[k]-> n_params();

        if (t_size != params.size()) {
          throw length_error("Wrong parameter vector size");
        }
        size_t iter = 0;
        for (size_t q = 0; q < A.size(); ++q) {
          if (A[q].size() <= 0)
            A[q] = mat(n_outputs, n_outputs, fill::zeros);
          for (size_t i = 0; i < A[q].n_rows; ++i) {
            for (size_t j = 0; j < A[q].n_cols; ++j) {
              A[q](i, j) = params[iter];
              if (j > i && fabs(params[iter]) > datum::eps) {
                throw logic_error("Params matrix must be lower triangular");
              }
              ++iter;
            }
          }
          B[q] = A[q] * A[q].t();
          if (!check_symmetric(B[q]))
            cout << "Matrix B is not symmetric :(" << endl;
        }

        for (size_t k = 0; k < kernels.size(); ++k) {
          vector<double> subparams(params.begin() + iter,
              params.begin() + iter + kernels[k]-> n_params());

          kernels[k]->set_params(subparams);
          iter += (kernels[k]-> n_params());
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
        for (size_t i = 0; i < B.size(); ++i)
          ans += B[i].size();

        for (size_t i = 0; i < kernels.size(); ++i)
          ans += kernels[i]->n_params();

        return ans;
      }

      void check_bounds(const vector<double> &bounds) {
        size_t iter = 0;
        for (size_t q = 0; q < A.size(); ++q) {
          for (size_t i = 0; i < A[q].n_rows; ++i) {
            for (size_t j = 0; j < A[q].n_cols; ++j) {
              if (j > i && fabs(bounds[iter]) > datum::eps)
                throw logic_error("Wrong bounds: Params matrix must be lower triangular");
              iter++;
            }
          }
        }
      }

      void set_lower_bounds(const vector<double> &lower_bound) {
        check_bounds(lower_bound);
        lower_bounds = lower_bound;
      }

      void set_upper_bounds(const vector<double> &upper_bound) {
        check_bounds(upper_bound);
        upper_bounds = upper_bound;
      }

      size_t in_kernel_np() {
        size_t ans = 0;
        for (size_t k = 0; k < kernels.size(); ++k) {
          ans += kernels[k]-> n_params();
        }
        return ans;
      }

      void set_lower_bounds(const double &l_bound) {
        vector<double> lower_bound(n_params(), l_bound);
        size_t iter = 0;
        for (size_t q = 0; q < A.size(); ++q) {
          for (size_t i = 0; i < A[q].n_rows; ++i) {
            for (size_t j = 0; j < A[q].n_cols; ++j) {
              if (j > i)
                lower_bound[iter] = 0.0;
              iter++;
            }
          }
        }
        lower_bounds = lower_bound;
      }

      void set_upper_bounds(const double &u_bound) {
        vector<double> upper_bound(n_params(), u_bound);
        size_t iter = 0;
        for (size_t q = 0; q < A.size(); ++q){
          for (size_t i = 0; i < A[q].n_rows; ++i){
            for (size_t j = 0; j < A[q].n_cols; ++j){
              if (j > i)
                upper_bound[iter] = 0.0;
              iter++;
            }
          }
        }
        upper_bounds = upper_bound;
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

    lmc_kernel::lmc_kernel(const size_t lf_number, size_t n_outputs) : lmc_kernel() {
      pimpl-> default_constructor(lf_number, n_outputs);
    }

    lmc_kernel::~lmc_kernel() {
      delete pimpl;
    }

    mat lmc_kernel::eval(const vector<mat> &X, const vector<mat> &Y) const {
      return pimpl-> eval(X, Y);
    }

    mat lmc_kernel::derivate(size_t param_id, const vector<mat> &X,
      const vector<mat> &Y) const {
      return pimpl-> derivate(param_id, X, Y);
    }

    mat lmc_kernel::diag_deriv(size_t param_id, const vector<mat> &X,
      const vector<mat> &Y) const{
      return pimpl-> derivate(param_id, X, Y, true);
    }

    size_t lmc_kernel::n_params() const {
      return pimpl-> n_params();
    }

    void lmc_kernel::set_params_k(const vector<mat> &params) {
      pimpl-> set_params_k(params);
    }

    void lmc_kernel::set_params(const vector<double> &params, size_t n_outputs) {
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
      return pimpl-> get_params();
    }

    vector<mat> lmc_kernel::get_params_k() const {
      return pimpl-> B;
    }

    vector<shared_ptr<kernel_class>> lmc_kernel::get_kernels() const {
      return pimpl-> kernels;
    }

    void lmc_kernel::set_lower_bounds(const double &lower_bounds) {
      pimpl-> set_lower_bounds(lower_bounds);
    }

    void lmc_kernel::set_upper_bounds(const double &upper_bounds) {
      pimpl-> set_upper_bounds(upper_bounds);
    }

    void lmc_kernel::set_lower_bounds(const vector<double> &lower_bounds) {
      pimpl-> set_lower_bounds(lower_bounds);
    }

    void lmc_kernel::set_upper_bounds(const vector<double> &upper_bounds) {
      pimpl-> set_upper_bounds(upper_bounds);
    }

    vector<double> lmc_kernel::get_lower_bounds() const {
      return pimpl-> lower_bounds;
    }

    vector<double> lmc_kernel::get_upper_bounds() const {
      return pimpl-> upper_bounds;
    }

  };
};
