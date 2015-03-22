#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {

    struct mv_gauss::implementation {
      vec mean;
      mat cov;
      mat cov_chol;
    };

    mv_gauss::mv_gauss() {
        pimpl = new implementation();
    }

    mv_gauss::mv_gauss(const vec& mean, const mat& cov) {
        pimpl = new implementation();
        pimpl->mean = mean;
        pimpl->cov = cov;
        pimpl->cov_chol = chol(cov);
    }

    mv_gauss::mv_gauss(const mv_gauss& other) {
        pimpl->mean = other.pimpl->mean;
        pimpl->cov = other.pimpl->cov;
        pimpl->cov_chol = other.pimpl->cov_chol;
    }

    vec mv_gauss::get_mean() const {
        return pimpl->mean;
    }

    void mv_gauss::set_mean(const vec& mean) {
        pimpl->mean = mean;
    }

    mat mv_gauss::get_cov() const {
        return pimpl->cov;
    }

    void mv_gauss::set_cov(const mat& cov) {
        pimpl->cov = cov;
        pimpl->cov_chol = chol(cov);
    }

    mat mv_gauss::get_cov_inv() const {
        mat tmp = upper_triangular_inverse(pimpl->cov_chol);
        return tmp * tmp.t();
    }

    size_t mv_gauss::dimension() const {
        return pimpl->mean.n_rows;
    }

    double mv_gauss::log_density(const arma::vec& x) const {
        int d = pimpl->mean.n_elem;
        double ans = -0.5 * d * log(2 * pi);
        //Now sum the log of the determinant of the covariance
        for (int i = 0; i < d; ++i)
            ans += -log(pimpl->cov_chol(i, i));
        mat sigm_inv = get_cov_inv();
        vec diff = x - pimpl->mean;
        ans += -0.5 * dot(diff, sigm_inv * diff);
        return ans;
    }

    double mv_gauss::density(const arma::vec& x) const {
        return exp(log_density(x));
    }

    mat mv_gauss::sample(int n_samples) const {
        size_t d = pimpl->mean.n_elem;
        mat ans = randn(n_samples, d) * pimpl->cov_chol;
        ans.each_row() += pimpl->mean.t();
        return ans;
    }

    mv_gauss mv_gauss::marginalize_hidden(const vector<bool>& observed) const {
        // myassert(observed.size() == mean.n_elem);
        vector<int> observed_ids;
        for (size_t i = 0; i < observed.size(); ++i)
            if (observed[i]) observed_ids.push_back(i);

        size_t n = observed_ids.size();
        vec new_mean(n);
        mat new_cov(n, n);
        for (size_t i = 0; i < n; ++i) {
            new_mean[i] = pimpl->mean[observed_ids[i]];
            for (size_t j = 0; j < n; ++j)
                new_cov(i, j) = pimpl->cov(observed_ids[i], observed_ids[j]);
        }
        return mv_gauss(new_mean, new_cov);
    }

    mv_gauss mv_gauss::conditional(const arma::vec &observation, const vector<bool> &observed) const {
        // myassert(observation.size() == mean.n_elem);
        // myassert(observed.size() == mean.n_elem);
        vector<unsigned int> v_obs_ix, v_hidden_ix;

        split_indices(observed, v_obs_ix, v_hidden_ix);

        uvec obs_ix(v_obs_ix), hidden_ix(v_hidden_ix);
        mat tmp = pimpl->cov(hidden_ix, obs_ix) * inv(pimpl->cov(obs_ix, obs_ix));
        vec new_mean = pimpl->mean(hidden_ix) + tmp * (observation(obs_ix) - pimpl->mean(obs_ix));
        mat new_cov  = pimpl->cov(hidden_ix, hidden_ix) - tmp * pimpl->cov(obs_ix, hidden_ix);
        return mv_gauss(new_mean, new_cov);
    }

    mv_gauss mv_gauss::operator=(const mv_gauss &other) {
        if (&other != this) {
            pimpl->mean = other.pimpl->mean;
            pimpl->cov = other.pimpl->cov;
            pimpl->cov_chol = other.pimpl->cov_chol;
        }
        return *this;
    }

};

/*
#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {

    mv_gauss::mv_gauss() {
    }

    mv_gauss::mv_gauss(const vec& mean, const mat& cov) {
        pimpl->mean = mean;
        pimpl->cov = cov;
        pimpl->cov_chol = chol(cov);
    }

    mv_gauss::mv_gauss(const mv_gauss& other) {
        pimpl->mean = other.mean;
        pimpl->cov = other.cov;
        pimpl->cov_chol = other.cov_chol;
    }

    vec mv_gauss::get_mean() const {
        return pimpl->mean;
    }

    void mv_gauss::set_mean(const vec& mean) {
        pimpl->mean = mean;
    }

    mat mv_gauss::get_cov() const {
        return pimpl->cov;
    }

    void mv_gauss::set_cov(const mat& cov) {
        pimpl->cov = cov;
        pimpl->cov_chol = chol(cov);
    }

    mat mv_gauss::get_cov_inv() const {
        mat tmp = upper_triangular_inverse(cov_chol);
        return tmp * tmp.t();
    }

    size_t mv_gauss::dimension() const {
        return mean.n_rows;
    }

    double mv_gauss::log_density(const arma::vec& x) const {
        int d = mean.n_elem;
        double ans = -0.5 * d * log(2 * pi);
        //Now sum the log of the determinant of the covariance
        for (int i = 0; i < d; ++i)
            ans += -log(cov_chol(i, i));
        mat sigm_inv = get_cov_inv();
        vec diff = x - mean;
        ans += -0.5 * dot(diff, sigm_inv * diff);
        return ans;
    }

    double mv_gauss::density(const arma::vec& x) const {
        return exp(log_density(x));
    }

    mat mv_gauss::sample(int n_samples) const {
        size_t d = mean.n_elem;
        mat ans = randn(n_samples, d) * cov_chol;
        ans.each_row() += mean.t();
        return ans;
    }

    mv_gauss mv_gauss::marginalize_hidden(const vector<bool>& observed) const {
        // myassert(observed.size() == mean.n_elem);
        vector<int> observed_ids;
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

    mv_gauss mv_gauss::conditional(const arma::vec &observation, const vector<bool> &observed) const {
        // myassert(observation.size() == mean.n_elem);
        // myassert(observed.size() == mean.n_elem);
        vector<unsigned int> v_obs_ix, v_hidden_ix;

        split_indices(observed, v_obs_ix, v_hidden_ix);

        uvec obs_ix(v_obs_ix), hidden_ix(v_hidden_ix);
        mat tmp = cov(hidden_ix, obs_ix) * inv(cov(obs_ix, obs_ix));
        vec new_mean = mean(hidden_ix) + tmp * (observation(obs_ix) - mean(obs_ix));
        mat new_cov  = cov(hidden_ix, hidden_ix) - tmp * cov(obs_ix, hidden_ix);
        return mv_gauss(new_mean, new_cov);
    }

    uuu    if (&other != pimpl) {
        if (&other != pimpl) {
            mean = other.mean;
            cov = other.cov;
            cov_chol = other.cov_chol;
        }
        return *pimpl;
    }

};*/
