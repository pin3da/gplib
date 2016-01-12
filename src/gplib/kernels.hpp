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
      /**
       * Squared Exponential class definition
       **/
      private:
        struct implementation;
        implementation *pimpl;
      public:

        /**
         * Constructor.
         **/
        squared_exponential();

        /**
         *  Constructor with params.
         *  @param params : Temporal
         **/
        squared_exponential(const std::vector<double> &params);
        /**
         * Destructor
         **/
        ~squared_exponential();
        /**
         *  @param X : Temporal.
         *  @param Y : Temporal.
         *  @param diag : Temporal.
         **/
        arma::mat eval(const arma::mat &X, const arma::mat &Y,
          bool diag = false) const;
        /**
         *  @param param_id : Temporal.
         *  @param X : Temporal.
         *  @param Y : Temporal.
         *  @param diag : Temporal.
         **/
        arma::mat derivate(size_t param_id, const arma::mat &X,
          const arma::mat &Y, bool diag = false) const;
        /**
         *  Temporal.
         **/
        size_t n_params() const;
        /**
         *  @param params : Temporal.
         **/
        void set_params(const std::vector<double> &params);
        /**
         *  @param lower_bounds : Temporal.
         **/
        void set_lower_bounds(const std::vector<double> &lower_bounds);
        /**
         *  @param upper_bounds : Temporal.
         **/
        void set_upper_bounds(const std::vector<double> &upper_bounds);
        /**
         *  Temporal.
         **/
        std::vector<double> get_params() const;
        /**
         *  Temporal.
         **/
        std::vector<double> get_lower_bounds() const;
        /**
         *  Temporal.
         **/
        std::vector<double> get_upper_bounds() const;
    };
  }
}

#endif
