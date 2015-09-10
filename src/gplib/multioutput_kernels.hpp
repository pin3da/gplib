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
          const std::vector<arma::mat> &Y, size_t id_out_1, size_t id_out_2) const;
        void set_params(const std::vector<arma::mat> &params);
        void set_param(size_t q, size_t a, size_t b, const double param);
        void set_param(size_t q, size_t param_id, const double param);
        void set_kernels(const std::vector<std::shared_ptr<kernel_class>> &kernels);
        std::vector<arma::mat> get_params() const;
        std::vector<std::shared_ptr<kernel_class>> get_kernels() const;
        double get_param(size_t q, size_t a, size_t b) const;
        double get_param(size_t q, size_t param_id) const;
    };

  }
}

#endif
