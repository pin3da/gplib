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
       * Squared Exponential Class definition
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
         *  Constructor, requires the hiperparameter.
         *  @param params : Vector of hiperparameters
         **/
        squared_exponential(const std::vector<double> &params);
        /**
         * Destructor
         **/
        ~squared_exponential();
        /**
         *  Evaluates the kernel function over the provided matrices.
         *  @param X : First matrix for kernel evaluation.
         *  @param Y : Second matrix for kernel evaluation.
         *  @param diag : Flag, if it is true the kernel should only be evaluated
         *                for the entries pertaining to the diagonal of the answer
         *                matrix, this is due to performance reasons while using
         *                FITC.
         **/
        arma::mat eval(const arma::mat &X, const arma::mat &Y,
          bool diag = false) const;
        /**
         *  Returns the value of the derivative wrt a certain parameter with a
         *  a particular pair of input matrices.
         *  @param param_id : Identifier of the parameter we are derivating with
         *                    respect to.
         *  @param X : First matrix for derivative evaluation.
         *  @param Y : Second matrix for derivative evaluation.
         *  @param diag : Flag, if it is true the kernel should only be evaluated
         *                for the derivative entries pertaining to the diagonal of
         *                the answer matrix, this is due to performance reasons
         *                while using FITC.
         **/
        arma::mat derivate(size_t param_id, const arma::mat &X,
          const arma::mat &Y, bool diag = false) const;
        /**
         *  Returns the number of params needed by the kernel.
         **/
        size_t n_params() const;
        /**
         *  Sets the parameters of the kernel using the proided vector
         *  @param params : vector containing all the parameters needed by the
         *  kernel.
         **/
        void set_params(const std::vector<double> &params);
        /**
         *  Sets the lower bounds to be used by the kernel during training process
         *  @param lower_bounds : Vector containing the lower bounds to be used.
         **/
        void set_lower_bounds(const std::vector<double> &lower_bounds);
        /**
         *  Sets the upper bounds to be used by the kernel during training process
         *  @param upper_bounds : Vector containing the upper bounds to be used.
         **/
        void set_upper_bounds(const std::vector<double> &upper_bounds);
        /**
         *  Returns a vector with the current values of the parameters of the
         *  kernel.
         **/
        std::vector<double> get_params() const;
        /**
         *  Returns a vector with the current values of the lower_bounds for each
         *  of the parameters of the kernel.
         **/
        std::vector<double> get_lower_bounds() const;
        /**
         *  Returns a vector with the current values of the upper_bounds for each
         *  of the parameters of the kernel.
         **/
        std::vector<double> get_upper_bounds() const;
    };
  }
}

#endif
