#ifndef GPLIB_KERNEL
#define GPLIB_KERNEL

#include "gp.hpp"

namespace gplib {
  namespace kernels {
    class squared_exponential : public kernel_class {
     /**
      * \brief squared exponential kernel with noise inference
      *
      * This kernel is defined as sig ^ 2 * exp(- ((x - xp) * (x - xp)')/ 2 * l) + sig_noise ^ 2 * I
      *
      * @param params : vector of hyperparameters 0 : sig, 1 : l (length scale), 2 : sig_noise.
      * */
      private:
        struct implementation;
        implementation *pimpl;
      public:
        squared_exponential();
        squared_exponential(const std::vector<double> &params);
        ~squared_exponential();
        arma::mat eval(const arma::mat &X, const arma::mat &Y, size_t id_out_1,
            size_t id_out_2) const;
        arma::mat derivate(size_t param_id, const arma::mat &X, const arma::mat &Y,
            size_t id_out_1, size_t id_out_2) const;
        size_t n_params() const;
        void set_params(const std::vector<double> &params);
        void set_lower_bounds(const std::vector<double> &lower_bounds);
        void set_upper_bounds(const std::vector<double> &upper_bounds);
        std::vector<double> get_params() const;
        std::vector<double> get_lower_bounds() const;
        std::vector<double> get_upper_bounds() const;
    };
  }
}

#endif
