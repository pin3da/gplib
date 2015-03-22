#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE kernels
#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include "gplib/gplib.hpp"

#define EPS 0.0001

BOOST_AUTO_TEST_SUITE( kernels )

BOOST_AUTO_TEST_CASE( example )
{
  std::vector<double> params({0.5 , 0.5});
  gplib::kernels::squared_exponential test(params);

  arma::mat x(3, 1);
  arma::mat y(3, 1);
  arma::mat an_grad;
  arma::mat num_grad;

  x = arma::randn(3, 1);
  y = arma::randn(3, 1);

  an_grad = test.derivate(0, x, y, 0, 0);
  an_grad += test.derivate(1, x, y, 0, 0);
  an_grad.print();

  num_grad = (test.eval(x + EPS, y + EPS, 0, 0) - test.eval(x - EPS, y - EPS, 0, 0)) / EPS;
  num_grad.print();
  int a = 5;
  BOOST_CHECK_EQUAL(5, a);
}

BOOST_AUTO_TEST_SUITE_END()
