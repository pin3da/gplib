#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE kernels
#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include "gplib/kernels.hpp"

BOOST_AUTO_TEST_SUITE( kernels )

BOOST_AUTO_TEST_CASE( example )
{
  std::vector<double> params;
  params.push_back(0.5);
  params.push_back(0.5);

  gplib::kernels::squared_exponential test(params);
  /*gplib::kernels::squared_exponential test;
  test = gplib::kernels::squared_exponential(params);

  arma::mat x(3, 3);
  arma::mat y(3, 3);*/


  int a = 5;
  BOOST_CHECK_EQUAL(5, a);
}

BOOST_AUTO_TEST_SUITE_END()
