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
         *  Constructor
         *  @param kernels : Kernels used in the latent functions.
         *  @param params  : Matrix of params used in the lmc, 'b' matrix in @ref.
         **/
        lmc_kernel(const std::vector<std::shared_ptr<kernel_class>> &kernels,
            const std::vector<arma::mat> &params);

        /**
         *  Constructor with default kernels and parameters.
         *  @param lf_number : Number of latent functions.
         *  @param n_outputs : Number of outputs.
         **/
        lmc_kernel(const size_t lf_number, size_t n_outputs);

        /**
         *  Destructor
         **/
        ~lmc_kernel();

        /**
         *  @param X : Temporal.
         *  @param Y : Temporal.
         *  @param diag : Temporal.
         **/
        arma::mat eval(const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y, bool diag = false) const;
        /**
         *  @param param_id : Temporal.
         *  @param X : Temporal.
         *  @param Y : Temporal.
         *  @param diag : Temporal.
         **/
        arma::mat derivate(size_t param_id, const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y, bool diag = false) const;
        /**
         *  Temporal
         **/
        size_t n_params() const;
        /**
         *  @param params : Temporal.
         **/
        void set_params_k(const std::vector<arma::mat> &params);
        /**
         *  @param params : Temporal.
         *  @param n_outputs : Temporal.
         **/
        void set_params(const std::vector<double> &params,
          size_t n_outputs = 0);
        /**
         *  @param q : Temporal.
         *  @param a : Temporal.
         *  @param b : Temporal.
         *  @param param : Temporal.
         **/
        void set_param(size_t q, size_t a, size_t b, const double param);
        /**
         *  @param q : Temporal.
         *  @param param_id : Temporal.
         *  @param param : Temporal.
         **/
        void set_param(size_t q, size_t param_id, const double param);
        /**
         *  @param kernels : Temporal.
         **/
        void set_kernels(const std::vector<std::shared_ptr<kernel_class>>
          &kernels);
        /**
         *  Temporal.
         **/
        std::vector<arma::mat> get_params_k() const;
        /**
         *  Temporal.
         **/
        std::vector<double> get_params() const;
        /**
         *  Temporal.
         **/
        std::vector<std::shared_ptr<kernel_class>> get_kernels() const;
        /**
         *  Temporal.
         *  @param q : Temporal.
         *  @param a : Temporal.
         *  @param b : Temporal.
         **/
        double get_param(size_t q, size_t a, size_t b) const;
        /**
         *  Temporal.
         *  @param q : Temporal.
         *  @param param_id : Temporal.
         **/
        double get_param(size_t q, size_t param_id) const;
        /**
         *  @param lower_bounds : Temporal.
         **/
        void set_lower_bounds(const double &lower_bounds);
        /**
         *  @param upper_bounds : Temporal.
         **/
        void set_upper_bounds(const double &upper_bounds);
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
        std::vector<double> get_lower_bounds() const;
        /**
         *  Temporal.
         **/
        std::vector<double> get_upper_bounds() const;
    };

  }
}

#endif
