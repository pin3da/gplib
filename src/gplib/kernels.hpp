#ifndef GPLIB_KERNEL
#define GPLIB_KERNEL

#include "gp.hpp"

namespace gplib {
  namespace kernels {

   /**
    * Squared exponential kernel with noise inference.
    *
    * This kernel is defined as sig ^ 2 * exp(- ((x - xp) * (x - xp)')/ 2 * l) + sig_noise ^ 2 * I
    *
    * @note
    *   params : vector of hyperparameters 0 : sig, 1 : l (length scale), 2 : sig_noise.
    */
    class squared_exponential : public kernel_class {
      private:
        struct implementation;
        implementation *pimpl;
      public:

        /**
         * Test doc for constructor.
         */
        squared_exponential();

        /**
         * Test doc for constructor with params.
         */
        squared_exponential(const std::vector<double> &params);
        ~squared_exponential();
        arma::mat eval(const arma::mat &X, const arma::mat &Y) const;
        arma::mat derivate(size_t param_id, const arma::mat &X, const arma::mat &Y) const;
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
