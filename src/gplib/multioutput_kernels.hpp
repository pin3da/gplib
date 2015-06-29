#ifndef GPLIB_MULTIOUTPUT_KERNEL
#define GPLIB_MULTIOUTPUT_KERNEL

#include "gp.hpp"

namespace gplib{
  namespace multioutput_kernels{
    class lmc_kernel : public multioutput_kernel_class {
      private:
        struct implementation;
        implementation *pimpl;
      public:
        lmc_kernel();
        lmc_kernel(const std::vector<std::shared_ptr<kernel_class>> &kernels, const std::vector<arma::mat> &params);
        ~lmc_kernel();

        arma::mat eval(const std::vector<arma::mat> &X, unsigned int lf_number);
        arma::mat derivate();
        void set_params(const std::vector<arma::mat> &params);
        void set_kernels(const std::vector<std::shared_ptr<kernel_class>> &kernels);
        std::vector<arma::mat> get_params();
        std::vector<std::shared_ptr<kernel_class>> get_kernels();
    };

  }
}

#endif
