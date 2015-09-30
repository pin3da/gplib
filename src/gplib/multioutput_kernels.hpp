#ifndef GPLIB_MULTIOUTPUT_KERNEL
#define GPLIB_MULTIOUTPUT_KERNEL

#include "gp.hpp"

namespace gplib{
  namespace multioutput_kernels{
    /**
     * Linear model of coregionalization
     * @ref : http://www.jmlr.org/papers/volume12/alvarez11a/alvarez11a.pdf
     * */
    class lmc_kernel : public multioutput_kernel_class {
      private:
        struct implementation;
        implementation *pimpl;
      public:
        lmc_kernel();
        /**
         *  @param kernels : Kernels used in the latent functions.
         *  @param params  : Matrix of params used in the lmc, 'b' matrix in @ref.
         * */
        lmc_kernel(const std::vector<std::shared_ptr<kernel_class>> &kernels,
            const std::vector<arma::mat> &params);
        ~lmc_kernel();

        arma::mat eval(const std::vector<arma::mat> &X, const std::vector<arma::mat> &Y) const ;
        arma::mat derivate(size_t param_id, const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y) const;
        size_t n_params() const;
        void set_params_k(const std::vector<arma::mat> &params);
        void set_params(const std::vector<double> &params, size_t n_outputs = -1);
        void set_param(size_t q, size_t a, size_t b, const double param);
        void set_param(size_t q, size_t param_id, const double param);
        void set_kernels(const std::vector<std::shared_ptr<kernel_class>> &kernels);
        std::vector<arma::mat> get_params_k() const;
        std::vector<double> get_params() const;
        std::vector<std::shared_ptr<kernel_class>> get_kernels() const;
        double get_param(size_t q, size_t a, size_t b) const;
        double get_param(size_t q, size_t param_id) const;
        void set_lower_bounds(const double &lower_bounds);
        void set_upper_bounds(const double &upper_bounds);
        void set_lower_bounds(const std::vector<double> &lower_bounds);
        void set_upper_bounds(const std::vector<double> &upper_bounds);
        std::vector<double> get_lower_bounds() const;
        std::vector<double> get_upper_bounds() const;
    };

  }
}

#endif
