#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {

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

};
