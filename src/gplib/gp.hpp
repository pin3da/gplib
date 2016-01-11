
#ifndef GPLIB_GP
#define GPLIB_GP

#include <armadillo>
#include <vector>
#include <memory>
#include <cassert>

#include "mvgauss.hpp"

namespace gplib {

    class kernel_class {
    /* *
     * Kernel Class definition
     * */
    public:
      /* *
       * Constructor
       * */
      kernel_class() {};
      /* *
       * Destructor
       * */
      virtual ~kernel_class() = default;
      /* *
       *  @param X : Temporal.
       *  @param Y : Temporal.
       *  @param diag : Temporal.
       * */
      virtual arma::mat eval(const arma::mat &X, const arma::mat &Y,
          bool diag = false) const = 0;
      /* *
       *  @param param_id : Temporal.
       *  @param X : Temporal.
       *  @param Y : Temporal.
       *  @param diag : Temporal.
       * */
      virtual arma::mat derivate(size_t param_id, const arma::mat &X,
          const arma::mat &Y, bool diag = false) const = 0;
      /* *
       *  Temporal.
       * */
      virtual size_t n_params() const = 0;
      /* *
       *  @param params : Temporal.
       * */
      virtual void set_params(const std::vector<double> &params) = 0;
      /* *
       *  @param lower_bounds : Temporal.
       * */
      virtual void set_lower_bounds(const std::vector<double> &lower_bounds) = 0;
      /* *
       *  @param upper_bounds : Temporal.
       * */
      virtual void set_upper_bounds(const std::vector<double> &upper_bounds) = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<double> get_params() const = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<double> get_lower_bounds() const = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<double> get_upper_bounds() const = 0;
    };

    class gp_reg {
    /* *
     * GP Regression Class definition
     * */
    private:
      struct implementation;
      implementation* pimpl;
    public:
      /* *
       * Constructor
       * */
      gp_reg();
      /* *
       * Destructor
       * */
      ~gp_reg();
      /* *
       *  @param k : Temporal.
       * */
      void set_kernel(const std::shared_ptr<kernel_class> &k);
      /* *
       *  Temporal.
       * */
      std::shared_ptr<kernel_class> get_kernel() const;
      /* *
       *  @param X : Temporal.
       *  @param y : Temporal.
       * */
      void set_training_set(const arma::mat &X, const arma::vec &y);
      /* *
       *  @param max_iter : Temporal.
       *  @param tol : Temporal.
       * */
      double train(int max_iter, double tol);
      /* *
       *  @param new_data : Temporal.
       * */
      mv_gauss full_predict(const arma::mat &new_data) const;
      /* *
       *  @param new_data : Temporal.
       * */
      arma::vec predict(const arma::mat &new_data) const;
    };

    class multioutput_kernel_class {
    /* *
     * Multioutput Kernel Class definition
     * */
    public:
      /* *
       * Constructor
       * */
      multioutput_kernel_class () {};
      /* *
       *  Constructor
       *  @param kernels : Temporal.
       *  @param params : Temporal.
       * */
      multioutput_kernel_class (const std::vector<std::shared_ptr<kernel_class>> &kernels,
            const std::vector<arma::mat> &params) {}
      /* *
       * Destructor
       * */
      virtual ~multioutput_kernel_class() = default;
      /* *
       *  @param X : Temporal.
       *  @param Y : Temporal.
       *  @param diag : Temporal.
       * */
      virtual arma::mat eval(const std::vector<arma::mat> &X,
        const std::vector<arma::mat> &Y, bool diag = false) const = 0;
      /* *
       *  @param param_id : Temporal.
       *  @param X : Temporal.
       *  @param Y : Temporal.
       *  @param diag : Temporal.
       * */
      virtual arma::mat derivate(size_t param_id, const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y, bool diag = false) const = 0;
      /* *
       *  Temporal.
       * */
      virtual size_t n_params() const = 0;
      /* *
       *  @param params : Temporal.
       * */
      virtual void set_params_k(const std::vector<arma::mat> &params) = 0;
      /* *
       *  @param params : Temporal.
       *  @param n_outputs : Temporal.
       * */
      virtual void set_params(const std::vector<double> &params, size_t n_outputs = -1) = 0;
      /* *
       *  @param kernels : Temporal.
       * */
      virtual void set_kernels(const std::vector<std::shared_ptr<kernel_class>> &kernels) = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<arma::mat> get_params_k() const = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<double> get_params() const = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<std::shared_ptr<kernel_class>> get_kernels() const = 0;
      /* *
       *  @param lower_bounds : Temporal.
       * */
      virtual void set_lower_bounds(const double &lower_bounds) = 0;
      /* *
       *  @param upper_bounds : Temporal.
       * */
      virtual void set_upper_bounds(const double &upper_bounds) = 0;
      /* *
       *  @param lower_bounds : Temporal.
       * */
      virtual void set_lower_bounds(const std::vector<double> &lower_bounds) = 0;
      /* *
       *  @param params : Temporal.
       * */
      virtual void set_upper_bounds(const std::vector<double> &params) = 0;
      /* *
       * Temporal.
       * */
      virtual std::vector<double> get_lower_bounds() const = 0;
      /* *
       *  Temporal.
       * */
      virtual std::vector<double> get_upper_bounds() const = 0;
      /* *
       *  Temporal.
       * */
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
      /* *
       * Constructor
       * */
      gp_reg_multi();
      /* *
       * Destructor
       * */
      ~gp_reg_multi();
      /* *
       *  @param k : Temporal.
       * */
      void set_kernel(const std::shared_ptr<multioutput_kernel_class> &k);
      /* *
       *  @param max_iter : Temporal.
       *  @param tol : Temporal.
       * */
      double train(const int max_iter, const double tol);
      /* *
       *  @param max_iter : Temporal.
       *  @param tol : Temporal.
       *  @param num_pi : Temporal.
       * */
      double train(const int max_iter, const double tol, const size_t num_pi);
      /* *
       *  @param max_iter : Temporal.
       *  @param tol : Temporal.
       *  @param num_pi : Temporal.
       * */
      double train(const int max_iter, const double tol,
        const std::vector<size_t> num_pi);
      /* *
       *  @param max_iter : Temporal.
       *  @param tol : Temporal.
       *  @param M : Temporal.
       * */
      double train(const int max_iter, const double tol,
        const std::vector<arma::mat> num_pi);
      /* *
       *  @param X : Temporal.
       *  @param y : Temporal.
       * */
      void set_training_set(const std::vector<arma::mat> &X,
        const std::vector<arma::vec> &y);
      /* *
       *  @param new_data : Temporal.
       * */
      mv_gauss full_predict(const std::vector<arma::mat> &new_data);
      /* *
       *  @param new_data : Temporal.
       * */
      arma::vec predict(const std::vector<arma::mat> &new_data) const;
      /* *
       *  Temporal.
       * */
      std::vector<double> get_params() const;
      /* *
       *  @param params : Temporal.
       * */
      void set_params(const std::vector<double> &params);
      enum {FULL, FITC};
    };
};

#endif
