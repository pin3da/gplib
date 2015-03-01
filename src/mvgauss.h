
#ifndef GPLIB_MVGAUSS
#define GPLIB_MVGAUSS

#include <armadillo>
#include <vector>

namespace gplib {

    //Multivariate Gaussian Distribution
    class MVGauss {
        arma::vec mean;
        arma::mat cov;
        arma::mat cov_chol;
    public:
        MVGauss();
        MVGauss(const MVGauss& other);
        MVGauss(const arma::vec& mean, const arma::mat& cov);
        arma::vec getMean() const;
        void setMean(const arma::vec& mean);
        arma::mat getCov() const;
        void setCov(const arma::mat& cov);
        arma::mat getCovInv() const;
        unsigned int dimension() const;

        /* Returns nSamples samples in a matrix with nSamples rows and D
         * columns. Where D is the dimensionality of the Gaussian distribution
         */
        arma::mat sample(int nSamples) const;

        double log_density(const arma::vec& x) const;
        double density(const arma::vec& x) const;

        /* Returns the marginal distribution after integrating out the
         * non observed variables. If the value of observed[i] is true then
         * the variable is assumed observed, otherwise it is integrated out.
         */
        MVGauss marginalizeHidden(const std::vector<bool>& observed) const;

        /* Returns the conditional distribution of the hidden variables given the
         * an observation of the observed variables. Only the values for which the
         * observed vector is true are considered on vector observation.
         */
        MVGauss conditional(const arma::vec& observation, const std::vector<bool>& observed) const;

        MVGauss operator=(const MVGauss& other);
    };
};

#endif
