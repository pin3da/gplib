#include "gplib/gplib.hpp"
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;
using namespace gplib;

int main() {
    double ang = 45.0*pi / 180.0;
    mat rot, scale;
    rot << cos(ang) << -sin(ang) << endr << sin(ang) << cos(ang) << endr;
    scale << 16 << 0 << endr << 0 << 4 << endr;
    vec mean {10,10};
    mat cov = rot*scale*rot.t();
    mv_gauss g(mean, cov);
    cout<<"2"<<endl;
    mat samples = g.sample(30);
    cout<<"3"<<endl;
    cout << "samples = " << endl;
    cout << samples << endl;

    mat cov_inv = g.get_cov_inv();
    cout << "cov = " << cov << endl << "covInv = " << cov_inv << endl;
    cout << "cov*cov_inv=" << cov_inv * cov << endl;
}
