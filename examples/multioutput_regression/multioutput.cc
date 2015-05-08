/*This is a very simple example of a single output
Gaussian Process Regression applied to a sin function*/

#include <gplib/gplib.hpp>
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;
using namespace gplib;

const int MN = 100;
const double m_pi = acos(-1);
#define __d { db(__LINE__); }

inline void db(int line){
  cout<<line<<endl;
}

int main(int argc, char **argv) {
  vec nd_mean {0};
  mat nd_cov ({0.001});
  srand (time(NULL));

  mat x(MN, 1);
  mat new_x(MN, 1);
  vec y(MN * 3);
  vec new_y(MN * 3);

  mv_gauss noise_dist(nd_mean, nd_cov);
  double j = 0.0, new_j;
  for (int i = 0; i < MN; i++, j+= 0.5) {
    double noise = noise_dist.sample(1)(0, 0);
    x(i, 0) = j;
    new_j = j + ((rand() % 10) + 1) / 25.0;
    new_x(i, 0) = new_j;
    y(i) = sin(j) + noise;
    //y(i) = j + noise;
    new_y(i) = sin(new_j) + noise;
    //new_y(i) = new_j + noise;

    y(i + MN) = sin(j + m_pi * 0.25) + noise;
    //y(i + MN) = j + 0.25 + noise;
    new_y(i + MN) = sin(new_j + m_pi * 0.25) + noise;
    //new_y(i + MN) = new_j + 0.25 + noise;

    y(i + 2*MN) = 0.3 * sin(j) + noise;
    //y(i + 2*MN) = 0.8 * j + noise;
    new_y(i + 2*MN) = 0.3 * sin(new_j) + noise;
    //new_y(i + 2*MN) = 0.8 * new_j + noise;
  }

  vector<mat> X_set(3), new_X_set(3);
  vector<double> kernel_params({0.1, 0.1, 0.05});
  vector<double> kernel1_params({1, 1, 0.05});
  //Set lower and upper bounds for parameters
  vector<double> lower_bounds({0.00001, 0.00001, 0.0001});
  vector<double> upper_bounds({5, 5, 0.1});
  /*we use a shared pointer so that the GPR class
  can handle the destruction and other management functions
  of the kernel*/
  auto sp_kernel = make_shared<kernels::squared_exponential>(kernel_params);
  sp_kernel->set_upper_bounds(upper_bounds);
  sp_kernel->set_lower_bounds(lower_bounds);

  auto sp_kernel1 = make_shared<kernels::squared_exponential>(kernel1_params);
  sp_kernel1->set_upper_bounds(upper_bounds);
  sp_kernel1->set_lower_bounds(lower_bounds);

  vector<shared_ptr<kernel_class> > kernels;
  //Create Regression object
  gp_reg_multi test_reg;

  for(int i = 0; i < 3; i++) {
    X_set[i] = x;
    new_X_set[i] = new_x;
  }
  mat params(X_set.size(), X_set.size());
  params.eye();
  test_reg.set_params(params);
  //Set the Kernel
  kernels.push_back(sp_kernel);
  kernels.push_back(sp_kernel1);
  test_reg.set_kernels(kernels);
  //Set training set as the generated Data (with noise)
  test_reg.set_training_set(X_set, y);
  //Take the posterior distribution for the new data
  mv_gauss posterior = test_reg.full_predict(new_X_set);
  //save training set
  x.save("x.mio", raw_ascii);
  y.save("y.mio", raw_ascii);
  //Save new datapoints and correct answers
  new_x.save("new_x.mio", raw_ascii);
  new_y.save("new_y.mio", raw_ascii);
  vec y_1 = new_y.subvec(0, MN - 1);
  y_1.save("new_y1.mio", raw_ascii);
  //Save results from GPR
  mat pos_cov = posterior.get_cov();
  cout << pos_cov.n_rows << endl;
  vec pos_mean = posterior.get_mean();

  vec mean1 = (pos_mean.subvec(0, MN - 1));
  mean1.save("mean1.mio", raw_ascii);

  vec mean2 = (pos_mean.subvec(MN, 2 * MN -1));
  mean2.save("mean2.mio", raw_ascii);

  vec mean3 = (pos_mean.subvec(2 * MN, 3 * MN - 1));
  mean3.save("mean3.mio", raw_ascii);
  pos_cov.save("cov.mio", raw_ascii);
  //posterior.get_mean().save("mean.mio", raw_ascii);
  return 0;

}
