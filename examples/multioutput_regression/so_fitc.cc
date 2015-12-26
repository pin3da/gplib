#include <cmath>
#include <iostream>
#include <armadillo>

#include <gplib/gplib.hpp>

using namespace std;
using namespace arma;
using namespace gplib;

const size_t MN = 50,
             noutputs = 1,
             l_functions = 1,
             num_pi = ceil(MN * 0.7),
             num_iter = 100;


const double m_pi = acos(-1);

int main(int argc, char **argv) {

  mat x(MN, 1);
  mat new_x(MN, 1);
  vector<vec> y(noutputs);
  vector<vec> new_y(noutputs);

  for(int i = 0; i < noutputs; i++) {
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
  }

  vector<mat> X_set(noutputs), new_X_set(noutputs);
  // Use same inputs for each output
  for(size_t i = 0; i < noutputs; i++) {
    X_set[i] = x;
    new_X_set[i] = new_x;
  }

  vector<double> kernel_params({0.1, 0.1, 0.05});

  vector<shared_ptr<kernel_class> > latent_functions;
  for(size_t i = 0; i < l_functions; i++) {
    latent_functions.push_back(
        make_shared<kernels::squared_exponential>(kernel_params));
  }

  vector<mat> params(latent_functions.size(), eye<mat>(noutputs, noutputs));
  auto K = make_shared<multioutput_kernels::lmc_kernel> (latent_functions, params);

  gp_reg_multi gp;
  K-> set_upper_bounds(1.0);
  K-> set_lower_bounds(0.01);
  gp.set_kernel(K);

  //Set training set as the generated Data (with noise)
  gp.set_training_set(X_set, y);
  gp.train(num_iter, gp_reg_multi::FITC, (void *) &num_pi);
  // gp.train(num_iter, gp_reg_multi::FULL, (void *) &num_pi);

  //Take the posterior distribution for the new data
  mv_gauss posterior = gp.full_predict(new_X_set);

  //save training set
  x.save("x.mio", raw_ascii);
  y[0].save("y.mio", raw_ascii);

  //Save new datapoints and correct answers
  new_x.save("new_x.mio", raw_ascii);
  new_y[0].save("new_y.mio", raw_ascii);

  //Save results from GPR
  mat pos_cov = posterior.get_cov();
  vec pos_mean = posterior.get_mean();
  vec mean1 = (pos_mean.subvec(0, MN - 1));
  mean1.save("mean.mio", raw_ascii);
  pos_cov.save("cov.mio", raw_ascii);
  return 0;
}
