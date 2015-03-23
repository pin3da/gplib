#include <gplib/gplib.hpp>
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;
using namespace gplib;

int main(int argc, char **argv) {
  vec noise_dist_mean {0};
  mat noise_dist_cov ({0.001});

  mat x(100, 1);
  vec y(100), new_y(100);
  mat new_x(100, 1);
  double noise = 0.0;
  double j = 0.0;
  mv_gauss noise_dist(noise_dist_mean, noise_dist_cov);
  for (int i = 0; i < 100; i++, j += 0.5){
    noise = noise_dist.sample(1)(0,0);
    x(i, 0) = j;
    y(i) = sin(j) + noise;
    new_x(i, 0) = j + 0.5; // ((rand() % 10) + 1) / 25.0;
    new_y(i) = sin(new_x(i, 0));
  }

  vector<double> kernel_params({0.5, 0.5});
  vector<double> lower_bounds({0, 0});
  vector<double> upper_bounds({5, 5});
  auto sp_kernel = make_shared<kernels::squared_exponential>(kernel_params);
  sp_kernel->set_upper_bounds(upper_bounds);
  sp_kernel->set_lower_bounds(lower_bounds);


  gp_reg test_reg;
  test_reg.set_kernel(sp_kernel);

  test_reg.set_training_set(x, y);
  test_reg.set_noise(0.001);

  test_reg.train(10000);

  mv_gauss posterior = test_reg.full_predict(new_x);

  int index = atoi(argv[1]);
  if (index == 0)
    new_x.print();
  if (index == 1)
      x.print();
  if (index == 2)
    new_y.print();
  if (index == 3)
    posterior.get_cov().print();
  if (index == 4)
    posterior.get_mean().print();
  return 0;

}
