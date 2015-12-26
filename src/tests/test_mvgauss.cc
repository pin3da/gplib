#include <boost/test/unit_test.hpp>
#include <iostream>
#include <armadillo>
#include <cmath>
#include <ctime>
#include <ratio>
#include <chrono>

#include "gplib/gplib.hpp"

using namespace std;

BOOST_AUTO_TEST_SUITE( mv_gauss )

BOOST_AUTO_TEST_CASE( mv_gauss_test ) {

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  double ang = 45.0 * gplib::pi / 180.0;
  arma::mat rot, scale;
  rot << cos(ang) << -sin(ang) << arma::endr << sin(ang) << cos(ang) << arma::endr;
  scale << 16 << 0 << arma::endr << 0 << 4 << arma::endr;
  arma::vec mean {10,10};
  arma::mat cov = rot*scale*rot.t();
  gplib::mv_gauss g(mean, cov);
  //cout<<"2"<<endl;
  arma::mat samples = g.sample(30);
  //cout<<"3"<<endl;
  //cout << "samples = " << endl;
  //cout << samples << endl;

  arma::mat cov_inv = g.get_cov_inv();
  //cout << "cov = " << cov << endl << "covInv = " << cov_inv << endl;
  //cout << "cov*cov_inv=" << cov_inv * cov << endl;

  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  std::cout << "\033[32m\t mv_gauss passed in "
      << time_span.count() << " seconds. \033[0m\n";

}

BOOST_AUTO_TEST_SUITE_END()
