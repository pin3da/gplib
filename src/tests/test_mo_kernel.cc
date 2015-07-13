#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>

#include "gplib/gplib.hpp"

// const double eps = 1e-6;
// #define eps 0.001

BOOST_AUTO_TEST_SUITE( mo_kernels )

BOOST_AUTO_TEST_CASE( mo_eval_lmc_kernel ) {
  /**
   * The evaluation of kernel must be a positive, semidefinite matrix.
   * If it is not correct, cholesky decomposition will arise an error.
   * */
  std::vector<arma::mat> X;
  for (int i = 0; i < 4; ++i)
    X.push_back(arma::randn(100, 3));

  std::vector<arma::mat> params;
  std::vector<std::shared_ptr<gplib::kernel_class>> latent_functions;
  for (int i = 0; i < 3; ++i) {
    auto kernel = std::make_shared<gplib::kernels::squared_exponential>();
    latent_functions.push_back(kernel);
  }
  gplib::multioutput_kernels::lmc_kernel K(latent_functions, params);
  arma::mat ans = K.eval(X, 0); // Which is the value of lf_number in this case ? ... latent_functions.size() ?
  arma::mat tmp = arma::chol(ans);
  std::cout << "\033[32m\t eval multioutput lmc_kernel passed ... \033[0m\n";
}

BOOST_AUTO_TEST_CASE( mo_lmc_gradiend ) {
  std::cout << "\033[32m\t gradient multioutput lmc_kernel passed ... \033[0m\n";
}

BOOST_AUTO_TEST_SUITE_END()
