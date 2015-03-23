#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE kernels
#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include "gplib/gplib.hpp"

#define EPS 0.0001

BOOST_AUTO_TEST_SUITE( kernels )

BOOST_AUTO_TEST_CASE( eval_kernel )
{
  /**
   * The evaluation of kernel must be a positive, semidefinite matrix.
   * If it is not correct, cholesky decomposition will arise an error.
   * */
  arma::mat X = arma::randn(4, 3);
  gplib::kernels::squared_exponential K(std::vector<double>({1.0, 2.3}));
  arma::mat ans = K.eval(X, X, 0, 0);
  arma::mat tmp = arma::chol(ans);
}

BOOST_AUTO_TEST_CASE( gradiend )
{
}

BOOST_AUTO_TEST_SUITE_END()
