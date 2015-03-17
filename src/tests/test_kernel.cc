#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE kernels
#include <boost/test/unit_test.hpp>

#include "gplib/kernels.hpp"

BOOST_AUTO_TEST_SUITE( kernels )

BOOST_AUTO_TEST_CASE( example )
{
  int a = 5;
  BOOST_CHECK_EQUAL(5, a);
}

BOOST_AUTO_TEST_SUITE_END()
