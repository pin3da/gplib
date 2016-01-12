#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {
  // TODO: move utilities here.
  mat upper_triangular_inverse(const mat& upper_t) {
    size_t d = upper_t.n_rows;
    //myassert(d == upper_t.n_cols);
    mat ans(d, d);
    ans.fill(0.0);
    vector<double> tmp(d);
    for (size_t i = 0; i < d; i++) {
      ans(i, i) = 1.0 / upper_t(i, i);
      for (size_t j = i + 1; j < d; j++)
        tmp[j] = upper_t(i, j) / upper_t(i, i);
      for (size_t j = i + 1; j < d; j++) {
        double factor = ans(i, j) = -tmp[j] / upper_t(j, j);
        for (size_t k = j+1; k < d; k++)
          tmp[k] += factor*upper_t(j,k);
      }
    }
    return ans;
  }

  arma::vec get_observed_only(const arma::vec& vec, const vector<bool>& observed) {
    //myassert(vec.n_elem == observed.size());
    vector<double> tmp;
    for (size_t i = 0; i < observed.size(); i++)
      if (observed[i])
        tmp.push_back(vec[i]);
    return arma::vec(tmp);
  }

  void split_indices(const vector<bool> &predicates, vector<arma::uword> &true_part, vector<arma::uword> &false_part) {
    for (size_t i = 0; i < predicates.size(); ++i) {
      if (predicates[i])
        true_part.push_back(i);
      else
        false_part.push_back(i);
    }
  }

  bool all_true(const vector<bool>& vec) {
    for (size_t i = 0; i < vec.size(); i++)
      if(!vec[i])
        return false;
    return true;
  }

  bool check_symmetric(const mat &A) {
    mat aux = A.t();
    for (size_t i = 0; i < A.n_rows; ++i)
      for (size_t j = 0; j < A.n_cols; ++j)
        if (A(i, j) != aux(i, j))
          return false;
    return true;

  }

  mat force_symmetric(const mat &A) {
    return (A + A.t()) / 2;
  }

  mat force_diag(const mat &A) {
    mat ans = A;
    double eps = 1e-6;
    for (size_t i = 0; i < A.n_rows;++i)
      if (fabs(ans(i, i)) < eps)
        ans(i, i) = eps;
    return ans;
  }

  bool is_close(const mat &A, const mat &B, double eps = 1e-4) {
    if (A.n_rows != B.n_rows || A.n_cols != B.n_cols)
      return false;

    for (size_t i = 0; i < A.n_rows; ++i)
      for (size_t j = 0; j < B.n_rows; ++j)
        if (fabs(A(i, j) - B(i, j)) > eps)
          return false;

    return true;
  }

  mat flatten(vector<vec> &y) {
    mat flat;
    for (size_t i = 0; i < y.size(); i++) {
      flat = join_cols<mat> (flat, y[i]);
    }
    return flat;
  }

  vector<double> flatten(vector<mat> &M) {
    size_t t_size = 0;
    for (size_t i = 0; i < M.size(); ++i)
      t_size += M[i].size();

    vector<double> ans(t_size);
    size_t iter = 0;
    for (size_t q = 0; q < M.size(); ++q) {
      copy (M[q].begin(), M[q].end(), ans.begin() + iter);
      iter += M[q].size();
    }
    return ans;
  }

  vector<mat> unflatten(vector<double> &M_params, vector<mat> &M) {
    size_t iter = 0;
    vector<mat> out(M.size());
    for(size_t q = 0; q < out.size(); ++q) {
      out[q].resize(M[q].n_rows, M[q].n_cols);
      for(size_t i = 0; i < M[q].n_rows; ++i)
        for(size_t j = 0; j < M[q].n_cols; ++j) {
          out[q](i, j) = M_params[iter];
          ++iter;
        }
    }
    return out;
  }

  void split(const vector<double> &theta, vector<double> &kernel_params,
      vector<double> &M_params) {
    copy(theta.begin(), theta.begin() + kernel_params.size(),
        kernel_params.begin());

    size_t k_size = kernel_params.size();
    size_t m_size = M_params.size();
    copy(theta.begin() + k_size, theta.begin() + k_size + m_size,
        M_params.begin());
  }

 };
