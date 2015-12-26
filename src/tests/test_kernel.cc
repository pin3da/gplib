// Module definition should only be in one of the tests
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE gplib

#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include <ctime>
#include <ratio>
#include <chrono>

#include "gplib/gplib.hpp"

const double eps = 1e-5;
// #define eps 0.001
//
using namespace std;

BOOST_AUTO_TEST_SUITE( kernels )

BOOST_AUTO_TEST_CASE( eval_kernel ) {
  /**
   * The evaluation of kernel must be a positive, semidefinite matrix.
   * If it is not correct, cholesky decomposition will arise an error.
   * */


  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  arma::mat X = arma::randn(40, 13);
  gplib::kernels::squared_exponential K(std::vector<double>({1.0, 2.3, 0.1}));
  arma::mat ans = K.eval(X, X);
  arma::mat tmp = arma::chol(ans);


  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  std::cout << "\033[32m\t eval kernel passed in "
       << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_CASE( gradiend ) {

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  std::vector<double> params({0.9, 1.2, 0.1});
  gplib::kernels::squared_exponential test(params);

  arma::mat X = arma::randn(30, 2);
  arma::mat an_grad;
  arma::mat num_grad;

  for (size_t i = 0; i < params.size(); ++i) {
    an_grad = test.derivate(i, X, X);
    params[i] += eps;
    test.set_params(params);
    num_grad = test.eval (X, X);
    params[i] -= 2.0 * eps;
    test.set_params(params);
    num_grad -= test.eval (X, X);

    num_grad = num_grad / (2.0 * eps);

    for (size_t j = 0; j < num_grad.n_rows; ++j) {
      for (size_t k = 0; k < num_grad.n_cols; ++k) {
        BOOST_CHECK_CLOSE (num_grad (j , k), an_grad (j, k), eps);
      }
    }

    params[i] += eps;
    test.set_params(params);
  }

  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  std::cout << "\033[32m\t gradient kernel passed in "
       << time_span.count() << " seconds. \033[0m\n";

}



BOOST_AUTO_TEST_CASE( gradiend_wrt_dta ) {


  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  std::vector<double> params({0.9, 1.2, 0.1});
  gplib::kernels::squared_exponential test(params);



  arma::mat X = arma::randn(5, 2);
  arma::mat Y = arma::randn(3, 2);
  arma::mat an_grad;
  arma::mat num_grad;

  size_t param_id = params.size();

  for (size_t i = 0; i < Y.n_rows; ++i) {
    for (size_t j = 0; j < Y.n_cols; ++j) {
      an_grad = test.derivate(param_id, X, Y);
      Y(i, j) += eps;
      num_grad = test.eval(X, Y);
      Y(i, j) -= 2.0 * eps;
      num_grad -= test.eval (X, Y);
      Y(i, j) += eps;
      num_grad = num_grad / (2.0 * eps);

      for (size_t l = 0; l < num_grad.n_rows; ++l) {
        for (size_t n = 0; n < num_grad.n_cols; ++n) {
          BOOST_CHECK_CLOSE (num_grad (l, n), an_grad (l, n), eps);
        }
      }


      an_grad = test.derivate(param_id, Y, X);
      Y(i, j) += eps;
      num_grad = test.eval(Y, X);
      Y(i, j) -= 2.0 * eps;
      num_grad -= test.eval (Y, X);
      Y(i, j) += eps;
      num_grad = num_grad / (2.0 * eps);

      for (size_t l = 0; l < num_grad.n_rows; ++l) {
        for (size_t n = 0; n < num_grad.n_cols; ++n) {
          BOOST_CHECK_CLOSE (num_grad (l, n), an_grad (l, n), eps);
        }
      }

      an_grad = test.derivate(param_id, Y, Y);
      Y(i, j) += eps;
      num_grad = test.eval(Y, Y);
      Y(i, j) -= 2.0 * eps;
      num_grad -= test.eval (Y, Y);
      Y(i, j) += eps;
      num_grad = num_grad / (2.0 * eps);

      for (size_t l = 0; l < num_grad.n_rows; ++l) {
        for (size_t n = 0; n < num_grad.n_cols; ++n) {
          BOOST_CHECK_CLOSE (num_grad (l, n), an_grad (l, n), eps);
        }
      }

      param_id++;
    }
  }

  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  std::cout << "\033[32m\t gradient wrt data kernel passed in "
       << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_SUITE_END()
