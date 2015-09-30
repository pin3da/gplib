#ifndef GPLIB_BASIC
#define GPLIB_BASIC

/* Include the basic files for this machine learning library */

#include <cmath>
#include <map>
#include <vector>

namespace gplib {
  //definition of basic constants
  const double pi = std::acos(-1);

  arma::mat upper_triangular_inverse(const arma::mat& upper_t);

  /* Takes a vector of real values and a boolean vector telling which dimensions are observed
   * and returns a new vector with the observed dimensions only.
   */
  arma::vec get_observed_only(const arma::vec& vec, const std::vector<bool>& observed);

  /*
   * Splits the indices (Zero indexed) on the ones where the pradicate is true and the part it is false
   */
  void split_indices(const std::vector<bool> &predicates, std::vector<arma::uword> &true_part, std::vector<arma::uword> &false_part);

  /* Return true if all the values in the boolean vector are true. */
  bool all_true(const std::vector<bool>& vec);
};

#endif
