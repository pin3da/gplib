#ifndef GPLIB_BASIC
#define GPLIB_BASIC

/* Include the basic files for this machine learning library */

#include <cmath>
#include <map>
#include <vector>

namespace gplib {
  //definition of basic constants
  const double pi = std::acos(-1);

  arma::mat upper_triangular_inverse(const arma::mat &upper_t);

  /**
   * Takes a vector of real values and a boolean vector telling which
   * dimensions are observed and returns a new vector with the
   * observed dimensions only.
   */
  arma::vec get_observed_only(const arma::vec &vec,
                              const std::vector<bool> &observed);

  /**
   * Splits the indices (zero indexed) on the ones where the pradicate
   * is true and the part it is false
   */
  void split_indices(const std::vector<bool> &predicates,
                     std::vector<arma::uword> &true_part,
                     std::vector<arma::uword> &false_part);

  /**
   * Return true if all the values in the boolean vector are true.
   **/
  bool all_true(const std::vector<bool> &vec);


  /**
   * Checks if the matrix is symmetric
   * */
  bool check_symmetric(const arma::mat &A);

  /**
   * Returns a new matrix which is (A + A.t()) / 2.0
   * A must be square.
   * */
  arma::mat force_symmetric(const arma::mat &A);

  /**
   * Returns a new matrix with no zeroes in the diagonal.
   * */
  arma::mat force_diag(const arma::mat &A);

  /**
   * Returns a vector-like matrix with all the values contained in y concatenated.
   * */
  arma::mat flatten(std::vector<arma::vec> &y);

  /**
   * Returns a vector with all the values contained in M concatenated.
   * */
  std::vector<double> flatten(std::vector<arma::mat> &M);

  /**
   * Returns a vector of matrices with all the values in M_params.
   * Each matrix of the vector has the dimensions provided by M.
   * */
  std::vector<arma::mat> unflatten(std::vector<double> &M_params,
                                   std::vector<arma::mat> &M);

  /**
   * Splits the params contained in 'theta' based on the size of kernel_params
   * and M_params.
   * */
  void split(const std::vector<double> &theta, std::vector<double> &kernel_params,
      std::vector<double> &M_params);
};

#endif
