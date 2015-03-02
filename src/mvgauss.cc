#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {

    mv_gauss::mv_gauss() {
    }

    mv_gauss::mv_gauss(const vec& mean, const mat& cov) {
        this->mean = mean;
        this->cov = cov;
        this->cov_chol = chol(cov);
    }

    mv_gauss::mv_gauss(const mv_gauss& other) {
        this->mean = other.mean;
        this->cov = other.cov;
        this->cov_chol = other.cov_chol;
    }

    vec mv_gauss::get_mean() const {
        return this->mean;
    }

    void mv_gauss::set_mean(const vec& mean) {
        this->mean = mean;
    }

    mat mv_gauss::get_cov() const {
        return this->cov;
    }

    void mv_gauss::set_cov(const mat& cov) {
        this->cov = cov;
        this->cov_chol = chol(cov);
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
        for (size_t i = 0; i < N; ++i) {
            new_mean[i] = mean[observed_ids[i]];
            for (size_t j = 0; j < N; ++j)
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

    mv_gauss mv_gauss::operator=(const mv_gauss &other) {
        if (&other != this) {
            mean = other.mean;
            cov = other.cov;
            cov_chol = other.cov_chol;
        }
        return *this;
    }

};
