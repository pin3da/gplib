#ifndef GPLIB_MULTIOUTPUT_KERNEL
#define GPLIB_MULTIOUTPUT_KERNEL

#include "gp.hpp"

namespace gplib{
  namespace multioutput_kernels{
    /**
     * Linear model of coregionalization
     * @ref : http://www.jmlr.org/papers/volume12/alvarez11a/alvarez11a.pdf
     **/
    class lmc_kernel : public multioutput_kernel_class {
      /**
       *  LMC Kernel Class definition
       **/
      private:
        struct implementation;
        implementation *pimpl;
      public:
        /**
         *  Constructor
         **/
        lmc_kernel();
        /**
         *  Constructor, requires the inner kernels to be used and the parameter
         *  matrices.
         *  @param kernels : Shared pointer containing a vector with the Inner
         *                   kernels (kernel_class).
         *  @param params : Vector containing the parameter matrices.
         **/
        lmc_kernel(const std::vector<std::shared_ptr<kernel_class>> &kernels,
            const std::vector<arma::mat> &params);

        /**
         *  Constructor, requires the number of latent functions and the number
         *  of outputs to be use, creates default kernels and parameters.
         *  @param lf_number : size_t, Number of latent functions.
         *  @param n_outputs : size_t, Number of outputs.
         **/
        lmc_kernel(const size_t lf_number, size_t n_outputs);

        /**
         *  Destructor
         **/
        ~lmc_kernel();

        /**
         *  Evaluates the kernel function over the provided sets of matrices.
         *  @param X : First vector of matrices for kernel evaluation.
         *  @param Y : Second vector of matrices for kernel evaluation.
         *  @param diag : Flag, if it is true the kernel should only be evaluated
         *                for the entries pertaining to the diagonal of the answer
         *                matrix, this is due to performance reasons while using
         *                FITC.
         **/
        arma::mat eval(const std::vector<arma::mat> &X,
            const std::vector<arma::mat> &Y, bool diag = false) const;
        /**
         *  Returns the value of the derivative wrt a certain parameter with a
         *  a particular pair of input matrices.
         *  @param param_id : Identifier of the parameter we are derivating with
         *                    respect to.
         *  @param X : First vector of matrices for derivative evaluation.
         *  @param Y : Second vector of matrices for derivative evaluation.
         *  @param diag : Flag, if it is true the kernel should only be evaluated
         *                for the derivative entries pertaining to the diagonal of
         *                the answer matrix, this is due to performance reasons
         *                while using FITC.
         **/
        arma::mat derivate(size_t param_id, const std::vector<arma::mat> &X,
            const std::vector<arma::mat> &Y, bool diag = false) const;
        /**
         *  Return the total number of parameters needed bythe kernel (parameter
         *  matrices, plus the parameters of each inner kernel).
         **/
        size_t n_params() const;
        /**
         *  Sets the parameters of the multioutput kernel only, doesn't affect the
         *  parameters of the inner kernels.
         *  @param params : Vector of matrices containing the parameters the size
         *                  of the vector should be the same as the number of
         *                  latent functions (inner kernels).
         **/
        void set_params_k(const std::vector<arma::mat> &params);
        /**
         *  Sets all the parameters of the multioutput kernel including those of
         *  the inner kernels using a std. vector.
         *  @param params : vector containing all the parameters needed by the
         *                  multioutput kernel.
         *  @param n_outputs : If passed a number bigger than 0 the parameter
         *                     matrices will be shaped according to the number
         *                     received.
         **/
        void set_params(const std::vector<double> &params,
            size_t n_outputs = 0);
        /**
         *  Sets the parameter of the matrix B in multioutput kernel.
         *  @param q: size_t, identifier of the latent function.
         *  @param a: size_t, identifier of the parameter row.
         *  @param b: size_t, identifier of the parameter col.
         *  @param param: double, the value of the parameter to be set.
         **/
        void set_param(size_t q, size_t a, size_t b, const double param);
        /**
         *  Sets the parameter of the matrix B in multioutput kernel.
         *  @param q: size_t, identifier of the inner kernel.
         *  @param param_id: size_t, identifier of the parameter.
         *  @param param: double, the value of the parameter to be set.
         **/
        void set_param(size_t q, size_t param_id, const double param);
        /**
         *  Sets the inner kernels.
         *  @param kernels : Shared pointer containing a vector of kernel_class
         *                   kernels.
         **/
        void set_kernels(const std::vector<std::shared_ptr<kernel_class>>
            &kernels);
        /**
         *  Returns a vector of matrices containing the parameters of the
         *  multioutput kernel, but not those of the inner kernels (in other words
         *  only the parameter matrices).
         **/
        std::vector<arma::mat> get_params_k() const;
        /**
         *  Returns a std. vector containing all of the parameters of the
         *  multioutput kernel, including those of each inner kernel.
         **/
        std::vector<double> get_params() const;
        /**
         *  Returns a Shared pointer to a vector containing the inner kernels
         *  currently set.
         **/
        std::vector<std::shared_ptr<kernel_class>> get_kernels() const;
        /**
         *  Returns a double with a single parameter of the matrix B of the
         *  multioutput kernel.
         *  @param q: size_t, identifier of the latent function.
         *  @param a: size_t, identifier of the param row.
         *  @param b: size_t, identifier of the param col.
         **/
        double get_param(size_t q, size_t a, size_t b) const;
        /**
         *  Returns a double with a single parameter in the inner
         *  kernel of the  multioutput kernel.
         *  @param q: size_t, identifier of the inner kernel.
         *  @param param_id: size_t, the identifier of the parameter.
         **/
        double get_param(size_t q, size_t param_id) const;
        /**
         *  Sets the lower bounds to be used by the kernel during training process
         *  including those of the inner kernels.
         *  @param lower_bounds : double, the lower bounds of all the parameters
         *  will be set to this value.
         **/
        void set_lower_bounds(const double &lower_bounds);
        /**
         *  Sets the upper bounds to be used by the kernel during training process
         *  including those of the inner kernels.
         *  @param upper_bounds : double, the upper bounds of all the parameters
         *  will be set to this value.
         **/
        void set_upper_bounds(const double &upper_bounds);
        /**
         *  Sets the lower bounds to be used by the kernel during training process
         *  including thos of the inner kernels.
         *  @param lower_bounds : Vector containing the lower bounds to be used.
         **/
        void set_lower_bounds(const std::vector<double> &lower_bounds);
        /**
         *  Sets the upper bounds to be used by the kernel during training process
         *  including thos of the inner kernels.
         *  @param upper_bounds : Vector containing the upper bounds to be used.
         **/
        void set_upper_bounds(const std::vector<double> &upper_bounds);
        /**
         * Returns a vector with the lower bounds of all the parameters, including
         * those of the inner kernels.
         **/
        std::vector<double> get_lower_bounds() const;
        /**
         * Returns a vector with the upper bounds of all the parameters, including
         * those of the inner kernels.
         **/
        std::vector<double> get_upper_bounds() const;
    };

  }
}

#endif
