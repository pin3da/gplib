
#ifndef GPLIB_GP
#define GPLIB_GP

#include <armadillo>
#include <vector>
#include <memory>
#include <cassert>

#include "mvgauss.hpp"

namespace gplib {

    class kernel_class {
    /**
     * Kernel Class definition
     * */
    public:
      kernel_class() {};
      virtual ~kernel_class() = default;
      /**
       *  @param X : Temporal.
       *  @param Y : Temporal.
       * */
      virtual arma::mat eval(const arma::mat &X, const arma::mat &Y,
          bool diag = false) const = 0;
      virtual arma::mat derivate(size_t param_id, const arma::mat &X,
          const arma::mat &Y, bool diag = false) const = 0;
      virtual size_t n_params() const = 0;
      virtual void set_params(const std::vector<double> &params) = 0;
      virtual void set_lower_bounds(const std::vector<double> &lower_bounds) = 0;
      virtual void set_upper_bounds(const std::vector<double> &params) = 0;
      virtual std::vector<double> get_params() const = 0;
      virtual std::vector<double> get_lower_bounds() const = 0;
      virtual std::vector<double> get_upper_bounds() const = 0;
    };

    class gp_reg {
    private:
      struct implementation;
      implementation* pimpl;
    public:
      gp_reg();
      ~gp_reg();
      void set_kernel(const std::shared_ptr<kernel_class> &k);
      std::shared_ptr<kernel_class> get_kernel() const;
      void set_training_set(const arma::mat &X, const arma::vec &y);
      double train(int max_iter, double tol);
      mv_gauss full_predict(const arma::mat &new_data) const;
      arma::vec predict(const arma::mat &new_data) const;
    };

    class multioutput_kernel_class {
    public:
      multioutput_kernel_class () {};
      multioutput_kernel_class (const std::vector<std::shared_ptr<kernel_class>> &kernels,
            const std::vector<arma::mat> &params) {}
      virtual ~multioutput_kernel_class() = default;
      virtual arma::mat eval(const std::vector<arma::mat> &X,
        const std::vector<arma::mat> &Y, bool diag = false) const = 0;
      virtual arma::mat derivate(size_t param_id, const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y) const = 0;
      virtual size_t n_params() const = 0;
      virtual void set_params_k(const std::vector<arma::mat> &params) = 0;
      virtual void set_params(const std::vector<double> &params, size_t n_outputs = -1) = 0;
      virtual void set_kernels(const std::vector<std::shared_ptr<kernel_class>> &kernels) = 0;
      virtual std::vector<arma::mat> get_params_k() const = 0;
      virtual std::vector<double> get_params() const = 0;
      virtual std::vector<std::shared_ptr<kernel_class>> get_kernels() const = 0;
      virtual void set_lower_bounds(const double &lower_bounds) = 0;
      virtual void set_upper_bounds(const double &upper_bounds) = 0;
      virtual void set_lower_bounds(const std::vector<double> &lower_bounds) = 0;
      virtual void set_upper_bounds(const std::vector<double> &params) = 0;
      virtual std::vector<double> get_lower_bounds() const = 0;
      virtual std::vector<double> get_upper_bounds() const = 0;
      virtual arma::mat diag_deriv(size_t param_id, const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y) const = 0;
    };

    class gp_reg_multi {
    /**
     * Multioutput GP Regression.
     * @ref: www.gatsby.ucl.ac.uk/~snelson/thesis.pdf
     * */
    private:
      struct implementation;
      implementation* pimpl;
    public:
      gp_reg_multi();
      ~gp_reg_multi();
      void set_kernel(const std::shared_ptr<multioutput_kernel_class> &k);
      void set_training_set(const std::vector<arma::mat> &X, const std::vector<arma::vec> &y);
      double train(int max_iter, double tol, size_t type = 0, void *param = NULL);
      mv_gauss full_predict(const std::vector<arma::mat> &new_data);
      arma::vec predict(const std::vector<arma::mat> &new_data) const;
      std::vector<double> get_params() const;
      void set_params(const std::vector<double> &params);
      enum {FULL, FITC};
    };
};

#endif
