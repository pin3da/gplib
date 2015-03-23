#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE kernels
#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include "gplib/gplib.hpp"

const double eps = 1e-6;
// #define eps 0.001

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
  std::vector<double> params({0.9, 1.2});
  gplib::kernels::squared_exponential test(params);

  arma::mat X = arma::randn(3, 1);
  arma::mat an_grad;
  arma::mat num_grad;


  for (size_t i = 0; i < params.size(); ++i) {
    an_grad = test.derivate(i, X, X, 0, 0);

    params[i] += eps;
    test.set_params(params);
    num_grad = test.eval (X, X, 0, 0);
    params[i] -= 2.0 * eps;
    test.set_params(params);
    num_grad -= test.eval (X, X, 0, 0);

    num_grad = num_grad / (2.0 * eps);

    for (size_t j = 0; j < num_grad.n_rows; ++j) {
      for (size_t k = 0; k < num_grad.n_cols; ++k) {
        BOOST_CHECK_CLOSE (num_grad (j , k), an_grad (j, k), eps);
      }
    }

    params[i] += eps;
    test.set_params(params);
  }

}

BOOST_AUTO_TEST_SUITE_END()
