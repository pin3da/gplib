#ifndef GPLIB_MVGAUSS
#define GPLIB_MVGAUSS

#include <armadillo>
#include <vector>

namespace gplib {

  class mv_gauss {
    /**
    *  Multivariate Gaussian Distribution Class definition
    **/
    private:
      struct implementation;
      implementation* pimpl;
    public:
      /**
       *  Constructor
       **/
      mv_gauss();
      /**
       *  Constructor, requieres another Multivariate Gaussian distribution to
       *  be copied.
       *  @param other : Multivariate Gaussian.
       **/
      mv_gauss(const mv_gauss &other);
      /**
       *  Constructor, requieres a arma vector of means and a matrix of
       *  covariances.
       *  @param mean : Vector of means.
       *  @param cov : Matrix of covariance.
       **/
      mv_gauss(const arma::vec &mean, const arma::mat &cov);

      /**
       *  Destructor
       **/
      ~mv_gauss();

      /**
       *  Sets the mean vector to the Gaussian distribution.
       *  @param mean : Vector of means.
       **/
      void set_mean(const arma::vec &mean);
      /**
       *  Sets the covariance matrix to the Gaussian distribution.
       *  @param cov : Matrix of covanriance.
       **/
      void set_cov(const arma::mat &cov);
      /**
       *  Gets the mean vector.
       **/
      arma::vec get_mean() const;
      /**
       *  Gets the covanriance matrix.
       **/
      arma::mat get_cov() const;
      /**
       *  Returns the inverse of the covanriance matrix.
       **/
      arma::mat get_cov_inv() const;
      /**
       *  Returns the cholesky decomposition of the covanriance matrix.
       **/
      arma::mat get_cov_chol() const;
      /**
       *  Returns the dimensionality of the Gaussian ditribution.
       **/
      size_t dimension() const;

      /**
       *  Returns n_samples samples in a matrix with n_samples rows and D
       *  columns. Where D is the dimensionality of the Gaussian distribution.
       *  @param n_samples : number of samples.
       **/
      arma::mat sample(int n_samples) const;

      /**
       *  Returns the logartihm density of the Gaussian distribution
       *  @param x : Vector of random variables.
       **/
      double log_density(const arma::vec &x) const;
      /**
       *  Returns the density of the Gaussian ditribution
       *  @param x : Vector of random variables.
       **/
      double density(const arma::vec &x) const;

      /**
       *  Returns the marginal distribution after integrating out the
       *  non observed variables. If the value of observed[i] is true then
       *  the variable is assumed observed, otherwise it is integrated out.
       *  @param observed : boolean vector who indicate which values are observed.
       **/
      mv_gauss marginalize_hidden(const std::vector<bool> &observed) const;

      /**
       *  Returns the conditional distribution of the hidden variables given the
       *  an observation of the observed variables. Only the values for which the
       *  observed vector is true are considered on vector observation.
       *  @param observation : vector, indicate the observed values.
       *  @param observed : boolean vector, indicate which values are observed.
       **/
      mv_gauss conditional(const arma::vec &observation, const std::vector<bool> &observed) const;

      /**
       *  Overload of the operand = to work with Multivariate Gaussian class.
       *  @param other : Gaussian distribution to be set.
       **/
      mv_gauss operator=(const mv_gauss &other);
  };
};

#endif
