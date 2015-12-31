#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include <ctime>
#include <ratio>
#include <chrono>

#include "gplib/gplib.hpp"

const double eps = 1e-5;

using namespace std;

BOOST_AUTO_TEST_SUITE( mo_kernels )


BOOST_AUTO_TEST_CASE( mo_kernel_get_set ) {

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  const int noutputs = 4;

  vector<shared_ptr<gplib::kernel_class>> latent_functions;
  vector<double> ker_par({0.9, 1.2, 0.1});
  for (int i = 0; i < noutputs - 1; ++i) {
    auto kernel = make_shared<gplib::kernels::squared_exponential>(ker_par);
    latent_functions.push_back(kernel);
  }
  vector<arma::mat> params(latent_functions.size(),
                           arma::eye<arma::mat>(noutputs, noutputs));

  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);

  vector<size_t> movable;
  size_t cur = 0;
  for (size_t k = 0; k < latent_functions.size(); ++k) {
    for (size_t i = 0; i < noutputs; ++i) {
      for (size_t j = 0; j < noutputs; ++j) {
        if (j <= i) movable.push_back(cur);
        cur++;
      }
    }
  }

  for (size_t k = 0; k < latent_functions.size(); ++k)
    for (size_t i = 0; i < latent_functions[k]-> n_params(); ++i)
      movable.push_back(cur++);

  vector<double> p = K.get_params();
  size_t num_p = latent_functions.size() *
                 (noutputs * noutputs + ker_par.size());
  BOOST_CHECK_EQUAL(p.size(), num_p);

  for (size_t i = 0; i < movable.size(); ++i)
    p[movable[i]] = 10.0 / (random() % 10 + 1);

  K.set_params(p);

  vector<double> tmp = K.get_params();
  for (size_t i = 0; i < p.size(); ++i) {
    BOOST_CHECK_EQUAL(p[i], tmp[i]);
  }


  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "\033[32m\t get_set [multioutput lmc_kernel] passed in "
       << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_CASE( mo_eval_lmc_kernel ) {
  /**
   * The evaluation of kernel must be a positive, semidefinite matrix.
   * If it is not correct, cholesky decomposition will arise an error.
   * */

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  vector<arma::mat> X;
  const int noutputs = 4;
  for (int i = 0; i < noutputs; ++i)
    X.push_back(arma::randn(100, 3));

  vector<shared_ptr<gplib::kernel_class>> latent_functions;
  vector<double> ker_par({0.9, 1.2, 0.1});
  for (int i = 0; i < noutputs - 1; ++i) {
    auto kernel = make_shared<gplib::kernels::squared_exponential>(ker_par);
    latent_functions.push_back(kernel);
  }
  vector<arma::mat> params(latent_functions.size(),
                           arma::eye<arma::mat>(noutputs, noutputs));

  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);
  arma::mat ans = K.eval(X, X);
  arma::mat tmp = arma::chol(ans);


  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "\033[32m\t eval [multioutput lmc_kernel] passed in "
       << time_span.count() << " seconds. \033[0m\n";
}
BOOST_AUTO_TEST_CASE( mo_eval_diag_lmc_kernel ) {
  /**
   * The evaluation of kernel must be a positive, semidefinite matrix.
   * If it is not correct, cholesky decomposition will arise an error.
   * */

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  vector<arma::mat> X;
  const int noutputs = 4;
  for (int i = 0; i < noutputs; ++i)
    X.push_back(arma::randn(3, 1));

  vector<shared_ptr<gplib::kernel_class>> latent_functions;
  vector<double> ker_par({0.9, 1.2, 0.1});
  for (int i = 0; i < noutputs - 1; ++i) {
    auto kernel = make_shared<gplib::kernels::squared_exponential>(ker_par);
    latent_functions.push_back(kernel);
  }
  vector<arma::mat> params(latent_functions.size(),
                           arma::eye<arma::mat>(noutputs, noutputs));

  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);
  arma::mat ans = K.eval(X, X);
  arma::mat tmp = arma::diagmat(ans);
  arma::mat diag = K.eval(X, X, true);

  for (size_t i = 0; i < diag.n_rows; ++i )
    for (size_t j = 0; j < diag.n_cols; ++j)
      BOOST_CHECK_EQUAL(diag(i, j), tmp(i, j));


  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "\033[32m\t eval [multioutput lmc_kernel] passed in "
       << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_CASE( mo_lmc_gradient ) {

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  vector<arma::mat> X;
  const int noutputs = 4;
  const int n_points = 33;
  for (int i = 0; i < noutputs; ++i)
    X.push_back(arma::randn(n_points, 3));

  vector<shared_ptr<gplib::kernel_class>> latent_functions;
  vector<double> k_params({0.9, 1.2, 0.1});
  for (int i = 0; i < noutputs - 1; ++i) {
    auto kernel = make_shared<gplib::kernels::squared_exponential>(k_params);
    latent_functions.push_back(kernel);
  }
  vector<arma::mat> params(latent_functions.size(),
                           arma::eye<arma::mat>(noutputs, noutputs));

  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);
  int param_id = 0;
  arma::mat analitical;
  arma::mat numeric;
  //param_id = (q*d*d) + i * d + j for MO kernel
  //param_id = (Q * d * d) + q * params + i for normal kernel
  for (int k = 0; k < noutputs - 1; ++k) {
    for (int i = 0; i < noutputs; ++i) {
      for (int j = 0; j < noutputs; ++j) {
        param_id = (k * noutputs * noutputs) + i * noutputs + j;
        analitical = K.derivate(param_id, X, X);
        K.set_param(k, i, j, K.get_param(k, i, j) + eps);
        numeric = K.eval(X, X);
        K.set_param(k, i, j, K.get_param(k, i, j) - 2.0 * eps);
        numeric -= K.eval(X, X);
        numeric = numeric / (2.0 * eps);
        K.set_param(k, i, j, K.get_param(k, i, j) + eps);
        for (size_t l = 0; l < numeric.n_rows; ++l) {
          for (size_t n = 0; n < numeric.n_cols; ++n) {
            BOOST_CHECK_CLOSE (numeric (l, n), analitical (l, n), eps);
          }
        }
      }
    }
  }
  size_t offset = (latent_functions.size() * noutputs * noutputs);
  vector<double> lil_params;
  for (size_t i = 0; i < latent_functions.size(); ++i) {
    for (size_t j = 0; j < latent_functions[i] -> n_params(); ++j) {
      param_id = offset + i * latent_functions[i] -> n_params() + j;
      analitical = K.derivate(param_id, X, X);
      K.set_param(i, j, K.get_param(i, j) + eps);
      numeric = K.eval(X, X);
      K.set_param(i, j, K.get_param(i, j) - (2.0 * eps));
      numeric -= K.eval(X, X);
      numeric = numeric / (2.0 * eps);
      K.set_param(i, j, K.get_param(i, j) + eps);
      for (size_t l = 0; l < numeric.n_rows; ++l) {
        for (size_t n = 0; n < numeric.n_cols; ++n) {
          BOOST_CHECK_CLOSE (numeric (l, n), analitical (l, n), eps);
        }
      }
    }
  }


  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "\033[32m\t gradient [multioutput lmc_kernel] passed in "
       << time_span.count() << " seconds. \033[0m\n";

}

BOOST_AUTO_TEST_CASE( mo_lmc_gradient_wrt_data ) {
  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  srand(time(0));
  vector<arma::mat> X, Y;
  const double eps_close = 1e-4;

  // const size_t noutputs = random() % 5 + 2;
  const size_t noutputs = 4;

  for (size_t i = 0; i < noutputs; ++i) {
    X.push_back(arma::randn(30, 3));
    Y.push_back(arma::randn(20, 3));
  }

  vector<shared_ptr<gplib::kernel_class>> latent_functions;
  vector<double> k_params({0.9, 1.2, 0.1});
  for (size_t i = 0; i < noutputs - 1; ++i) {
    auto kernel = make_shared<gplib::kernels::squared_exponential>(k_params);
    latent_functions.push_back(kernel);
  }

  vector<arma::mat> params(latent_functions.size(),
                           arma::eye<arma::mat>(noutputs, noutputs));

  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);

  int param_id = latent_functions.size() * noutputs * noutputs +
                 k_params.size() * (noutputs - 1);

  arma::mat analitical;
  arma::mat numeric;

  for (size_t k = 0; k < noutputs; ++k) {
    for (size_t i = 0; i < Y[k].n_rows; ++i) {
      for (size_t j = 0; j < Y[k].n_cols; ++j) {
        analitical = K.derivate(param_id, X, Y);
        Y[k](i, j) += eps;
        numeric = K.eval(X, Y);
        Y[k](i, j) -= 2.0 * eps;
        numeric -= K.eval(X, Y);
        numeric = numeric / (2.0 * eps);
        Y[k](i, j) += eps;

        for (size_t l = 0; l < numeric.n_rows; ++l) {
          for (size_t n = 0; n < numeric.n_cols; ++n) {
            BOOST_CHECK_CLOSE(numeric (l, n), analitical (l, n), eps_close);
          }
        }

        analitical = K.derivate(param_id, Y, X);
        Y[k](i, j) += eps;
        numeric = K.eval(Y, X);
        Y[k](i, j) -= 2.0 * eps;
        numeric -= K.eval(Y, X);
        numeric = numeric / (2.0 * eps);
        Y[k](i, j) += eps;

        for (size_t l = 0; l < numeric.n_rows; ++l) {
          for (size_t n = 0; n < numeric.n_cols; ++n) {
            BOOST_CHECK_CLOSE(numeric (l, n), analitical (l, n), eps_close);
          }
        }

        // derivative of Kuu -> Both are inducing points
        analitical = K.derivate(param_id, Y, Y);
        Y[k](i, j) += eps;
        numeric = K.eval(Y, Y);
        Y[k](i, j) -= 2.0 * eps;
        numeric -= K.eval(Y, Y);
        numeric = numeric / (2.0 * eps);
        Y[k](i, j) += eps;

        for (size_t l = 0; l < numeric.n_rows; ++l) {
          for (size_t n = 0; n < numeric.n_cols; ++n) {
            BOOST_CHECK_CLOSE(numeric (l, n), analitical (l, n), eps_close);
          }
        }

        param_id++;
      }
    }
  }

  chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
  chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "\033[32m\t gradient wrt data [multioutput lmc_kernel] passed in " << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_CASE( mo_kernel_diag_deriv ) {
  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  srand(time(0));
  vector<arma::mat> X, Y;


  // const size_t noutputs = random() % 5 + 2;
  const size_t noutputs = 4;

  for (size_t i = 0; i < noutputs; ++i) {
    X.push_back(arma::randn(30, 3));
    Y.push_back(arma::randn(30, 3));
  }

  vector<shared_ptr<gplib::kernel_class>> latent_functions;
  vector<double> k_params({0.9, 1.2, 0.1});
  for (size_t i = 0; i < noutputs - 1; ++i) {
    auto kernel = make_shared<gplib::kernels::squared_exponential>(k_params);
    latent_functions.push_back(kernel);
  }

  vector<arma::mat> params(latent_functions.size(),
                           arma::eye<arma::mat>(noutputs, noutputs));

  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);

  arma::mat n_diag;
  arma::mat diag_deriv;
  for (size_t i = 0; i < K.n_params(); ++i){
    n_diag = diagmat(K.derivate(i, X, Y));
    diag_deriv = K.diag_deriv(i, X, Y);
    for (size_t l = 0; l < n_diag.n_rows; ++l)
      for (size_t n = 0; n < n_diag.n_cols; ++n)
        BOOST_CHECK_CLOSE(n_diag(l, n), diag_deriv(l, n), eps);
  }
  size_t param_id = K.n_params();
  for (size_t k = 0; k < noutputs; ++k){
    for (size_t i = 0; i < X[k].n_rows; ++i){
      for (size_t j = 0; j < X[k].n_cols; ++j){
        n_diag = diagmat(K.derivate(param_id, X, Y));
        diag_deriv = K.diag_deriv(param_id, X, Y);
        param_id++;
        for (size_t l = 0; l < n_diag.n_rows; ++l)
          for (size_t n = 0; n < n_diag.n_cols; ++n)
            BOOST_CHECK_CLOSE(n_diag(l, n), diag_deriv(l, n), eps);
      }
    }
  }

  chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
  chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "\033[32m\t diagonal derivative [multioutput lmc_kernel] passed in " << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_SUITE_END()
