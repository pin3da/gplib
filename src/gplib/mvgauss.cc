#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {

  struct mv_gauss::implementation {
    vec mean;
    mat cov;
    mat cov_chol;

    double log_density(const arma::vec& x, const arma::mat& sigm_inv) {
      int d = mean.n_elem;
      double ans = -0.5 * d * log(2 * pi);
      //Now sum the log of the determinant of the covariance
      for (int i = 0; i < d; ++i)
        ans += -log(cov_chol(i, i));
      vec diff = x - mean;
      ans += -0.5 * dot(diff, sigm_inv * diff);
      return ans;
    }

    mat sample(int n_samples) {
      size_t d = mean.n_elem;
      mat ans = randn(n_samples, d) * cov_chol;
      ans.each_row() += mean.t();
      return ans;
    }

    mv_gauss marginalize_hidden(const vector<bool>& observed) {
      vector<size_t> observed_ids;
      for (size_t i = 0; i < observed.size(); ++i)
        if (observed[i]) observed_ids.push_back(i);

      size_t n = observed_ids.size();
      vec new_mean(n);
      mat new_cov(n, n);
      for (size_t i = 0; i < n; ++i) {
        new_mean[i] = mean[observed_ids[i]];
        for (size_t j = 0; j < n; ++j)
          new_cov(i, j) = cov(observed_ids[i], observed_ids[j]);
      }
      return mv_gauss(new_mean, new_cov);
    }

    mv_gauss conditional(const arma::vec &observation, const vector<bool> &observed) {
      vector<uword> v_obs_ix, v_hidden_ix;

      split_indices(observed, v_obs_ix, v_hidden_ix);

      uvec obs_ix(v_obs_ix), hidden_ix(v_hidden_ix);
      mat tmp = cov(hidden_ix, obs_ix) * inv(cov(obs_ix, obs_ix));
      vec new_mean = mean(hidden_ix) + tmp * (observation(obs_ix) - mean(obs_ix));
      mat new_cov  = cov(hidden_ix, hidden_ix) - tmp * cov(obs_ix, hidden_ix);
      return mv_gauss(new_mean, new_cov);
    }

  };


  mv_gauss::mv_gauss() {
    pimpl = new implementation();
  }

  mv_gauss::mv_gauss(const vec& mean, const mat& cov) : mv_gauss() {
    pimpl->mean = mean;
    pimpl->cov = cov;
    pimpl->cov_chol = chol(cov + 1e-6 * eye<mat>(cov.n_rows, cov.n_cols));
  }

  mv_gauss::mv_gauss(const mv_gauss& other) : mv_gauss() {
    pimpl->mean = other.get_mean();
    pimpl->cov = other.get_cov();
    pimpl->cov_chol = other.get_cov_chol();
  }

  mv_gauss::~mv_gauss() {
    delete pimpl;
  }

  void mv_gauss::set_mean(const vec& mean) {
    pimpl->mean = mean;
  }

  void mv_gauss::set_cov(const mat& cov) {
    pimpl->cov = cov;
    pimpl->cov_chol = chol(cov);
  }

  vec mv_gauss::get_mean() const {
    return pimpl->mean;
  }

  mat mv_gauss::get_cov() const {
    return pimpl->cov;
  }

  mat mv_gauss::get_cov_chol() const {
    return pimpl->cov_chol;
  }

  mat mv_gauss::get_cov_inv() const {
    mat tmp = upper_triangular_inverse(pimpl->cov_chol);
    return tmp * tmp.t();
  }

  size_t mv_gauss::dimension() const {
    return pimpl->mean.n_rows;
  }

  double mv_gauss::log_density(const arma::vec& x) const {
    mat sigm_inv = get_cov_inv();
    return pimpl->log_density(x, sigm_inv);
  }

  double mv_gauss::density(const arma::vec& x) const {
    return exp(log_density(x));
  }

  mat mv_gauss::sample(int n_samples) const {
    return pimpl->sample(n_samples);
  }

  mv_gauss mv_gauss::marginalize_hidden(const vector<bool>& observed) const {
    return pimpl->marginalize_hidden(observed);
  }

  mv_gauss mv_gauss::conditional(const arma::vec &observation, const vector<bool> &observed) const {
    return pimpl->conditional(observation, observed);
  }

  mv_gauss mv_gauss::operator=(const mv_gauss &other) {
    if (&other != this) {
      pimpl->mean = other.get_mean();
      pimpl->cov = other.get_cov();
      pimpl->cov_chol = other.get_cov_chol();
    }
    return *this;
  }

};
