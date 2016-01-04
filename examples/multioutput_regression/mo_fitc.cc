/*This is a very simple example of a single output
Gaussian Process Regression applied to a sin function*/

#include <gplib/gplib.hpp>
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;
using namespace gplib;

const size_t MN = 100,
             num_pi = ceil(MN * 0.23),
             iter = 1000;

const double tol = 1e-4;

int main(int argc, char **argv) {
  size_t noutputs = 3, l_functions = 2;
  vec nd_mean {0};
  mat nd_cov ({0.001});
  srand (time(NULL));

  mat x(MN, 1);
  mat new_x(MN, 1);
  vector<vec> y(noutputs);
  vector<vec> new_y(noutputs);

  for(int i = 0; i < noutputs; i++) {
    y[i].resize(MN);
    new_y[i].resize(MN);
  }

  mv_gauss noise_dist(nd_mean, nd_cov);
  double j = 0.0, new_j;
  for (size_t i = 0; i < MN; i++, j+= 0.5) {
    double noise = noise_dist.sample(1)(0, 0);
    x(i, 0) = j;
    new_j = j + ((rand() % 10) + 1) / 25.0;
    new_x(i, 0) = new_j;
    y[0](i) = sin(j) + noise;
    new_y[0](i) = sin(new_j) + noise;

    y[1](i) = sin(j + pi * 0.25) + noise;
    new_y[1](i) = sin(new_j + pi * 0.25) + noise;

    y[2](i) = 0.3 * sin(j) + noise;
    new_y[2](i) = 0.3 * sin(new_j) + noise;
  }

  vector<mat> X_set(noutputs), new_X_set(noutputs);
  vector<double> kernel_params({0.1, 0.1, 0.05});

  //Set lower and upper bounds for parameters
  //vector<double> lower_bounds({0.00001, 0.00001, 0.0001});
  //vector<double> upper_bounds({5, 5, 0.1});

  /*we use a shared pointer so that the GPR class
  can handle the destruction and other management functions
  of the kernel*/

  vector<shared_ptr<kernel_class> > latent_functions;
  for(size_t i = 0; i < l_functions; i++) {
    latent_functions.push_back(make_shared<kernels::squared_exponential>(kernel_params));
  }

  vector<mat> params(latent_functions.size(), eye<mat>(noutputs, noutputs));
  auto K = make_shared<multioutput_kernels::lmc_kernel> (latent_functions, params);


  //Create Regression object
  gp_reg_multi test_reg;
  K-> set_upper_bounds(1.0);
  K-> set_lower_bounds(-1.0);
  test_reg.set_kernel(K);
  for(size_t i = 0; i < noutputs; i++) {
    X_set[i] = x;
    new_X_set[i] = new_x;
  }

  //Set training set as the generated Data (with noise)
  test_reg.set_training_set(X_set, y);

  test_reg.train(iter, tol, gp_reg_multi::FITC, (void *) &num_pi);

  //Take the posterior distribution for the new data
  mv_gauss posterior = test_reg.full_predict(new_X_set);

  //save training set
  x.save("x.mio", raw_ascii);
  y[0].save("y1.mio", raw_ascii);
  y[1].save("y2.mio", raw_ascii);
  y[2].save("y3.mio", raw_ascii);

  //Save new datapoints and correct answers
  new_x.save("new_x.mio", raw_ascii);
  new_y[0].save("new_y1.mio", raw_ascii);
  new_y[1].save("new_y2.mio", raw_ascii);
  new_y[2].save("new_y3.mio", raw_ascii);

  //Save results from GPR
  mat pos_cov = posterior.get_cov();
  vec pos_mean = posterior.get_mean();
  vec mean1 = (pos_mean.subvec(0, MN - 1));
  mean1.save("mean1.mio", raw_ascii);
  vec mean2 = (pos_mean.subvec(MN, 2 * MN -1));
  mean2.save("mean2.mio", raw_ascii);
  vec mean3 = (pos_mean.subvec(2 * MN, 3 * MN - 1));
  mean3.save("mean3.mio", raw_ascii);
  pos_cov.save("cov.mio", raw_ascii);
  return 0;

}
