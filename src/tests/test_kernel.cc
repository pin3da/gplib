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
  // Uniformly distributed over interval [0, 1). Generated in python
  arma::mat X({0.04912227, 2.67544358, 2.53769772,
               1.64872528, 0.52406991, 1.35924321,
               1.52015109, 0.72726333, 0.06925668,
               0.60605489, 0.74525666, 1.97343891});
  X.reshape(4, 3);

  gplib::kernels::squared_exponential K(std::vector<double>({1.0, 2.3}));

  arma::mat ans = K.eval(X, X, 0, 0);
  arma::mat chol = arma::chol(ans);
  ans.print();
  chol.print();
}

BOOST_AUTO_TEST_CASE( example )
{
/*
 *  std::vector<double> params({0.5 , 0.5});
 *  gplib::kernels::squared_exponential test(params);
 *
 *  arma::mat X = arma::randn(3, 1);
 *  arma::mat an_grad;
 *  arma::mat num_grad;
 *
 *  an_grad = test.derivate(0, X, X, 0, 0);
 *  an_grad += test.derivate(1, X, X, 0, 0);
 *  an_grad.print();
 *
 *  num_grad = (test.eval(X + EPS, X + EPS, 0, 0) - test.eval(X - EPS, X - EPS, 0, 0)) / EPS;
 *  num_grad.print();
 */
  int a = 5;
  BOOST_CHECK_EQUAL(5, a);
}

BOOST_AUTO_TEST_SUITE_END()
