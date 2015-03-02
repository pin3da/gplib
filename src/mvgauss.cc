#include "gplib.h"

using namespace arma;
using namespace std;

namespace gplib {

    MVGauss::MVGauss() {
    }

    MVGauss::MVGauss(const vec& mean, const mat& cov) {
        this->mean = mean;
        this->cov = cov;
        this->cov_chol = chol(cov);
    }

    MVGauss::MVGauss(const MVGauss& other) {
        this->mean = other.mean;
        this->cov = other.cov;
        this->cov_chol = other.cov_chol;
    }

    vec MVGauss::getMean() const {
        return this->mean;
    }

    void MVGauss::setMean(const vec& mean) {
        this->mean = mean;
    }

    mat MVGauss::getCov() const {
        return this->cov;
    }

    void MVGauss::setCov(const mat& cov) {
        this->cov = cov;
        this->cov_chol = chol(cov);
    }

    mat MVGauss::getCovInv() const {
        mat tmp = upperTriangularInverse(cov_chol);
        return tmp * tmp.t();
    }

    unsigned int MVGauss::dimension() const {
        return mean.n_rows;
    }

    double MVGauss::log_density(const arma::vec& x) const {
        int D = mean.n_elem;
        double ans = -0.5*D*log(2*pi);
        //Now sum the log of the determinant of the covariance
        for (int i=0; i<D; i++)
            ans += -log(cov_chol(i,i));
        mat SigmInv = getCovInv();
        vec diff = x - mean;
        ans += -0.5*dot(diff, SigmInv*diff);
        return ans;
    }

    double MVGauss::density(const arma::vec& x) const {
        return exp(log_density(x));
    }

    mat MVGauss::sample(int nSamples) const {
        unsigned int D = mean.n_elem;
        mat ans = randn(nSamples,D)*cov_chol;
        ans.each_row() += mean.t();
        return ans;
    }

    MVGauss MVGauss::marginalizeHidden(const vector<bool>& observed) const {
        myassert(observed.size() == mean.n_elem);
        vector<int> observedIds;
        for (unsigned int i=0; i<observed.size(); i++) {
            if (observed[i]) observedIds.push_back(i);
        }
        unsigned int N = observedIds.size();
        vec new_mean(N);
        mat new_cov(N,N);
        for (unsigned int i=0; i<N; i++) {
            new_mean[i] = mean[observedIds[i]];
            for (unsigned int j=0; j<N; j++) {
                new_cov(i,j) = cov(observedIds[i],observedIds[j]);
            }
        }
        return MVGauss(new_mean, new_cov);
    }

    MVGauss MVGauss::conditional(const arma::vec& observation, const vector<bool>& observed) const {
        myassert(observation.size() == mean.n_elem);
        myassert(observed.size() == mean.n_elem);
        vector<unsigned int> vobsIx, vhiddenIx;
        splitIndices(observed, vobsIx, vhiddenIx);

        uvec obsIx(vobsIx), hiddenIx(vhiddenIx);
        mat tmp = cov(hiddenIx,obsIx) * inv(cov(obsIx, obsIx));
        vec newMean = mean(hiddenIx) + tmp*(observation(obsIx) - mean(obsIx));
        mat newCov = cov(hiddenIx, hiddenIx) - tmp*cov(obsIx, hiddenIx);
        return MVGauss(newMean, newCov);
    }

    MVGauss MVGauss::operator=(const MVGauss &other) {
        if (&other != this) {
            mean = other.mean;
            cov = other.cov;
            cov_chol = other.cov_chol;
        }
        return *this;
    }

};
