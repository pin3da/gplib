#include <gplib>
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

    MVGauss g(mean, cov);
    mat samples = g.sample(30);
    cout << "samples = " << endl;
    cout << samples << endl;

    mat covInv = g.getCovInv();
    cout << "cov = " << cov << endl << "covInv = " << covInv << endl;
    cout << "cov*covInv=" << covInv * cov << endl;
}
