#include <gplib/gplib.hpp>
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;
using namespace gplib;

const int MN = 100;
const double m_pi = acos(-1);

int main() {
  vec nd_mean {0};
  mat nd_cov ({0.001});
  srand (time(NULL));

  mat x(MN, 1);
  mat y(MN, 3);

  mv_gauss noise_dist(nd_mean, nd_cov);
  for (int i = 0; i < MN; i++) {
    double noise = noise_dist.sample(1)(0, 0);
    x(i, 0) = i;
    y(i, 0) = sin(i) + noise;
    y(i, 1) = sin(i + m_pi * 0.25) + noise;
    y(i, 2) = 0.3 * sin(i) + noise;
  }

  x.save("x.mio", raw_ascii);
  y.save("y.mio", raw_ascii);


  return 0;
}
