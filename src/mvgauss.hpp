
#ifndef GPLIB_MVGAUSS
#define GPLIB_MVGAUSS

#include <armadillo>
#include <vector>

namespace gplib {

    //Multivariate Gaussian Distribution
    class mv_gauss {
        arma::vec mean;
        arma::mat cov;
        arma::mat cov_chol;
    public:
        mv_gauss();
        mv_gauss(const mv_gauss& other);
        mv_gauss(const arma::vec& mean, const arma::mat& cov);
        arma::vec get_mean() const;
        void set_mean(const arma::vec& mean);
        arma::mat get_cov() const;
        void set_cov(const arma::mat& cov);
        arma::mat get_cov_inv() const;
        size_t dimension() const;

        /* Returns n_samples samples in a matrix with n_samples rows and D
         * columns. Where D is the dimensionality of the Gaussian distribution
         */
        arma::mat sample(int n_samples) const;

        double log_density(const arma::vec& x) const;
        double density(const arma::vec& x) const;

        /* Returns the marginal distribution after integrating out the
         * non observed variables. If the value of observed[i] is true then
         * the variable is assumed observed, otherwise it is integrated out.
         */
        mv_gauss marginalize_hidden(const std::vector<bool>& observed) const;

        /* Returns the conditional distribution of the hidden variables given the
         * an observation of the observed variables. Only the values for which the
         * observed vector is true are considered on vector observation.
         */
        mv_gauss conditional(const arma::vec& observation, const std::vector<bool>& observed) const;

        mv_gauss operator=(const mv_gauss& other);
    };
};

#endif
