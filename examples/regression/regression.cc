/*This is a very simple example of a single output
Gaussian Process Regression applied to a sin function*/

#include <gplib/gplib.hpp>
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;
using namespace gplib;

int main(int argc, char **argv) {
  /*We are going to use a Gaussian (Normal) Distribution
  to add noise to our data (in other words we are generating
  a normally distrubuted noise), the assumption that data
  has a normally distributed noise is important due to how
  Gaussian process regression works, here we create the mean
  and the convariance matrix of what is going to be our distribution
  for noise generation*/
  vec noise_dist_mean {0};
  mat noise_dist_cov ({0.001});
  srand (time(NULL));

  //We are going to store our data in this vectors/matrices
  mat x(100, 1);
  vec y(100), new_y(100);
  mat new_x(100, 1);
  double noise = 0.0;
  double j = 0.0;
  //Create our noise distribution
  mv_gauss noise_dist(noise_dist_mean, noise_dist_cov);
  //Generate Data
  for (int i = 0; i < 100; i++, j += 0.5){
    noise = noise_dist.sample(1)(0, 0);
    x(i, 0) = j;
    //Add noise to the sin function
    y(i) = sin(j) + noise;
    /*Create test data points, that is data points
    that were not included in the test cases, in order to
    test the regression*/
    new_x(i, 0) = j + ((rand() % 10) + 1) / 25.0;
    /*Save the correct answers of the new data points for
    comparison purposes*/
    new_y(i) = sin(new_x(i, 0));
  }

  /*Creating a kernel for the GPR, in this case
  we use the squared exponential kernel that receives
  three paramters*/
  vector<double> kernel_params({0.1, 0.1, 0.05});
  //Set lower and upper bounds for parameters
  vector<double> lower_bounds({0.00001, 0.00001, 0.0001});
  vector<double> upper_bounds({5, 5, 0.1});
  /*we use a shared pointer so that the GPR class
  can handle the destruction and other management functions
  of the kernel*/
  auto sp_kernel = make_shared<kernels::squared_exponential>(kernel_params);
  sp_kernel->set_upper_bounds(upper_bounds);
  sp_kernel->set_lower_bounds(lower_bounds);

  //Create Regression object
  gp_reg test_reg;
  //Set the Kernel
  test_reg.set_kernel(sp_kernel);

  //Set training set as the generated Data (with noise)
  test_reg.set_training_set(x, y);
  //Set the noise of the generated Data
  // test_reg.set_noise(0.1);
  //Train with a mximum of 10000 iterations
  test_reg.train(10000);
  //Take the posterior distribution for the new data
  mv_gauss posterior = test_reg.full_predict(new_x);

  //save training set
  x.save("x.mio", raw_ascii);
  y.save("y.mio", raw_ascii);
  //Save new datapoints and correct answers
  new_x.save("new_x.mio", raw_ascii);
  new_y.save("new_y.mio", raw_ascii);
  //Save results from GPR
  posterior.get_cov().save("cov.mio", raw_ascii);
  posterior.get_mean().save("mean.mio", raw_ascii);
  return 0;

}
