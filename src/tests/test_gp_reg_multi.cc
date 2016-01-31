
#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <vector>
#include <ctime>
#include <ratio>
#include <chrono>

#include "gplib/gplib.hpp"

const double eps = 1e-5;
const double m_pi = acos(-1);

using namespace std;
using namespace arma;

BOOST_AUTO_TEST_SUITE( gp_reg_multi )

BOOST_AUTO_TEST_CASE( gp_reg_multi_get_set ) {

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  size_t noutputs = 2, l_functions = noutputs - 1;
  size_t MN = 40;

  mat x(MN, 1);
  mat new_x(MN, 1);
  vector<vec> y(noutputs);
  vector<vec> new_y(noutputs);

  for (size_t i = 0; i < noutputs; i++) {
    y[i].resize(MN);
    new_y[i].resize(MN);
  }

  double j = 0.0, new_j;
  for (size_t i = 0; i < MN; i++, j+= 0.5) {
    x(i, 0) = j;
    new_j = j + ((rand() % 10) + 1) / 25.0;
    new_x(i, 0) = new_j;
    y[0](i) = sin(j);
    new_y[0](i) = sin(new_j);

    y[1](i) = sin(j + m_pi * 0.25);
    new_y[1](i) = sin(new_j + m_pi * 0.25);
  }

  vector<mat> X_set(noutputs), new_X_set(noutputs);
  vector<double> kernel_params({0.1, 0.1, 0.05});

  vector<shared_ptr<gplib::kernel_class> > latent_functions;
  for (size_t i = 0; i < l_functions; i++) {
    latent_functions.push_back(make_shared<gplib::kernels::squared_exponential>(kernel_params));
  }

  vector<mat> params(latent_functions.size(), eye<mat>(noutputs, noutputs));
  auto K = make_shared<gplib::multioutput_kernels::lmc_kernel> (latent_functions, params);


  gplib::gp_reg_multi test_reg;
  K-> set_upper_bounds(1.0);
  K-> set_lower_bounds(-1.0);
  test_reg.set_kernel(K);
  for (size_t i = 0; i < noutputs; i++) {
    X_set[i] = x;
    new_X_set[i] = new_x;
  }

  //Set training set as the generated Data (with noise)
  test_reg.set_training_set(X_set, y);
  size_t num_pi = 20;
  vector<size_t> vec_pi(2, 20);
  test_reg.train(1, 1, vec_pi, true);

  vector<double> test_params;

  test_params = test_reg.get_all_params();
  size_t M_size = num_pi * X_set.size() * X_set[0].n_cols;
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

  for (size_t i = 0; i < M_size + 1; ++i)
    movable.push_back(cur++);

  size_t num_p = (K-> get_params()).size() +  M_size + 1;
  BOOST_CHECK_EQUAL(test_params.size(), num_p);

  for (size_t i = 0; i < movable.size(); ++i)
    test_params[movable[i]] = 100.0 / (random() % 100 + 1);


  test_reg.set_params(test_params);
  vector<double> tmp = test_reg.get_all_params();
  for (size_t i = 0; i < test_params.size(); ++i) {
    BOOST_CHECK_EQUAL(test_params[i], tmp[i]);
  }


  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "\033[32m\t get_set [gp_reg_multi] passed in "
    << time_span.count() << " seconds. \033[0m\n";
}

BOOST_AUTO_TEST_CASE( gp_reg_multi_no_opt ) {

  chrono::high_resolution_clock::time_point t1 =
    chrono::high_resolution_clock::now();

  size_t noutputs = 2, l_functions = noutputs - 1;
  size_t MN = 40;

  mat x(MN, 1);
  mat new_x(MN, 1);
  vector<vec> y(noutputs);
  vector<vec> new_y(noutputs);

  for (size_t i = 0; i < noutputs; i++) {
    y[i].resize(MN);
    new_y[i].resize(MN);
  }

  double j = 0.0, new_j;
  for (size_t i = 0; i < MN; i++, j+= 0.5) {
    x(i, 0) = j;
    new_j = j + ((rand() % 10) + 1) / 25.0;
    new_x(i, 0) = new_j;
    y[0](i) = sin(j);
    new_y[0](i) = sin(new_j);

    y[1](i) = sin(j + m_pi * 0.25);
    new_y[1](i) = sin(new_j + m_pi * 0.25);
  }

  vector<mat> X_set(noutputs), new_X_set(noutputs);
  vector<double> kernel_params({0.1, 0.1, 0.05});

  vector<shared_ptr<gplib::kernel_class> > latent_functions;
  for (size_t i = 0; i < l_functions; i++) {
    latent_functions.push_back(make_shared<gplib::kernels::squared_exponential>(kernel_params));
  }

  vector<mat> params(latent_functions.size(), eye<mat>(noutputs, noutputs));
  auto K = make_shared<gplib::multioutput_kernels::lmc_kernel> (latent_functions, params);


  gplib::gp_reg_multi test_reg;
  K-> set_upper_bounds(1.0);
  K-> set_lower_bounds(-1.0);
  test_reg.set_kernel(K);
  for (size_t i = 0; i < noutputs; i++) {
    X_set[i] = x;
    new_X_set[i] = new_x;
  }

  //Set training set as the generated Data (with noise)
  test_reg.set_training_set(X_set, y);
  //size_t num_pi = 20;
  vector<size_t> vec_pi(2, 20);
  test_reg.train(1, 1, vec_pi);
  test_reg.predict(new_X_set);

  chrono::high_resolution_clock::time_point t2 =
    chrono::high_resolution_clock::now();

  chrono::duration<double> time_span =
    chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "\033[32m\t no_opt_eval [gp_reg_multi] passed in "
    << time_span.count() << " seconds. \033[0m\n";
}


BOOST_AUTO_TEST_SUITE_END()
