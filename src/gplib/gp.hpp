
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
     **/
    public:
      /**
       * Constructor
       **/
      kernel_class() {};
      /**
       * Destructor
       **/
      virtual ~kernel_class() = default;
      /**
       *  Evaluates the kernel function over the provided matrices.
       *  @param X : First matrix for kernel evaluation.
       *  @param Y : Second matrix for kernel evaluation.
       *  @param diag : Flag, if it is true the kernel should only be evaluated
       *                for the entries pertaining to the diagonal of the answer
       *                matrix, this is due to performance reasons while using
       *                FITC.
       **/
      virtual arma::mat eval(const arma::mat &X, const arma::mat &Y,
          bool diag = false) const = 0;
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
      virtual arma::mat derivate(size_t param_id, const arma::mat &X,
          const arma::mat &Y, bool diag = false) const = 0;
      /**
       *  Returns the number of params needed by the kernel.
       **/
      virtual size_t n_params() const = 0;
      /**
       *  Sets the parameters of the kernel using the proided vector
       *  @param params : vector containing all the parameters needed by the
       *  kernel.
       **/
      virtual void set_params(const std::vector<double> &params) = 0;
      /**
       *  Sets the lower bounds to be used by the kernel during training process
       *  @param lower_bounds : Vector containing the lower bounds to be used.
       **/
      virtual void set_lower_bounds(const std::vector<double> &lower_bounds) = 0;
      /**
       *  Sets the upper bounds to be used by the kernel during training process
       *  @param upper_bounds : Vector containing the upper bounds to be used.
       **/
      virtual void set_upper_bounds(const std::vector<double> &upper_bounds) = 0;
      /**
       *  Returns a vector with the current values of the parameters of the
       *  kernel.
       **/
      virtual std::vector<double> get_params() const = 0;
      /**
       *  Returns a vector with the current values of the lower_bounds for each
       *  of the parameters of the kernel.
       **/
      virtual std::vector<double> get_lower_bounds() const = 0;
      /**
       *  Returns a vector with the current values of the upper_bounds for each
       *  of the parameters of the kernel.
       **/
      virtual std::vector<double> get_upper_bounds() const = 0;
    };

    class gp_reg {
    /**
     * GP Regression Class definition
     **/
    private:
      struct implementation;
      implementation* pimpl;
    public:
      /**
       * Constructor
       **/
      gp_reg();
      /**
       * Destructor
       **/
      ~gp_reg();
      /**
       *  Sets the kernel to be used during the regression process.
       *  @param k : A kernel_class kernel.
       **/
      void set_kernel(const std::shared_ptr<kernel_class> &k);
      /**
       *  Returns the current kernel set to be used during the regression
       *  process.
       **/
      std::shared_ptr<kernel_class> get_kernel() const;
      /**
       *  Sets the training set to be used during the training process.
       *  @param X : Matrix of known inputs, each row represents one input.
       *  @param y : Vector of known outputs corresponding to the known inputs.
       **/
      void set_training_set(const arma::mat &X, const arma::vec &y);
      /**
       *  Trains the model using the provided training set
       *  @param max_iter : Maximum number of iterations.
       *  @param tol : Relative tolerance on the optimization parameters.
       **/
      double train(int max_iter, double tol);
      /**
       *  Uses the already trained model to predict output values for new
       *  inputs provided in the parameter, this method returns the complete
       *  multivariate gaussian distribution resulting from the regression
       *  process.
       *  @param new_data : A matrix containing points for which output data
       *                    is unknown.
       **/
      mv_gauss full_predict(const arma::mat &new_data) const;
      /**
       *  Uses the already trained model to predict output values for new
       *  inputs provided in the parameter, this method returns only the mean
       *  of the multivariate gaussian distribution resulting from the
       *  regression process, which is the "best guess" for each new input.
       *  @param new_data : A matrix containing points for which output data
       *                    is unknown.
       **/
      arma::vec predict(const arma::mat &new_data) const;
    };

    class multioutput_kernel_class {
    /**
     * Multioutput Kernel Class definition
     **/
    public:
      /**
       * Constructor
       **/
      multioutput_kernel_class () {};
      /**
       *  Constructor, requires the inner kernels to be used and the parameter
       *  matrices.
       *  @param kernels : Shared pointer containing a vector with the Inner
       *                   kernels (kernel_class).
       *  @param params : Parameter matrices.
       **/
      multioutput_kernel_class (
          const std::vector<std::shared_ptr<kernel_class>> &kernels,
          const std::vector<arma::mat> &params) {}
      /**
       * Destructor
       **/
      virtual ~multioutput_kernel_class() = default;
      /**
       *  Evaluates the kernel function over the provided sets of matrices.
       *  @param X : First vector of matrices for kernel evaluation.
       *  @param Y : Second vector of matrices for kernel evaluation.
       *  @param diag : Flag, if it is true the kernel should only be evaluated
       *                for the entries pertaining to the diagonal of the answer
       *                matrix, this is due to performance reasons while using
       *                FITC.
       **/
      virtual arma::mat eval(const std::vector<arma::mat> &X,
        const std::vector<arma::mat> &Y, bool diag = false) const = 0;
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
      virtual arma::mat derivate(size_t param_id, const std::vector<arma::mat> &X,
          const std::vector<arma::mat> &Y, bool diag = false) const = 0;
      /**
       *  Returns the total number of parameters needed bythe kernel (parameter
       *  matrices, plus the parameters of each inner kernel).
       **/
      virtual size_t n_params() const = 0;
      /**
       *  Sets the parameters of the multioutput kernel only, doesn't affect the
       *  parameters of the inner kernels.
       *  @param params : Vector of matrices containing the parameters the size
       *                  of the vector should be the same as the number of
       *                  latent functions (inner kernels).
       **/
      virtual void set_params_k(const std::vector<arma::mat> &params) = 0;
      /**
       *  Sets all the parameters of the multioutput kernel including those of
       *  the inner kernels using a std. vector.
       *  @param params : vector containing all the parameters needed by the
       *                  multioutput kernel.
       *  @param n_outputs : If passed a number bigger than 0 the parameter
       *                     matrices will be shaped according to the number
       *                     received.
       **/
      virtual void set_params(const std::vector<double> &params,
          size_t n_outputs = 0) = 0;
      /**
       *  Sets the inner kernels.
       *  @param kernels : Shared pointer containing a vector of kernel_class
       *                   kernels.
       **/
      virtual void set_kernels(
          const std::vector<std::shared_ptr<kernel_class>> &kernels) = 0;
      /**
       *  Returns a vector of matrices containing the parameters of the
       *  multioutput kernel, but not those of the inner kernels (in other words
       *  only the parameter matrices).
       **/
      virtual std::vector<arma::mat> get_params_k() const = 0;
      /**
       *  Returns a std. vector containing all of the parameters of the
       *  multioutput kernel, including those of each inner kernel.
       **/
      virtual std::vector<double> get_params() const = 0;
      /**
       *  Returns a Shared pointer to a vector containing the inner kernels
       *  currently set.
       **/
      virtual std::vector<std::shared_ptr<kernel_class>> get_kernels() const = 0;
      /**
       *  Sets the lower bounds to be used by the kernel during training process
       *  including those of the inner kernels.
       *  @param lower_bounds : double, the lower bounds of all the parameters
       *  will be set to this value.
       **/
      virtual void set_lower_bounds(const double &lower_bounds) = 0;
      /**
       *  Sets the upper bounds to be used by the kernel during training process
       *  including those of the inner kernels.
       *  @param upper_bounds : double, the upper bounds of all the parameters
       *  will be set to this value.
       **/
      virtual void set_upper_bounds(const double &upper_bounds) = 0;
      /**
       *  Sets the lower bounds to be used by the kernel during training process
       *  including thos of the inner kernels.
       *  @param lower_bounds : Vector containing the lower bounds to be used.
       **/
      virtual void set_lower_bounds(const std::vector<double> &lower_bounds) = 0;
      /**
       *  Sets the upper bounds to be used by the kernel during training process
       *  including thos of the inner kernels.
       *  @param upper_bounds : Vector containing the upper bounds to be used.
       **/
      virtual void set_upper_bounds(const std::vector<double> &params) = 0;
      /**
       * Returns a vector with the lower bounds of all the parameters, including
       * those of the inner kernels.
       **/
      virtual std::vector<double> get_lower_bounds() const = 0;
      /**
       * Returns a vector with the upper bounds of all the parameters, including
       * those of the inner kernels.
       **/
      virtual std::vector<double> get_upper_bounds() const = 0;
    };

    class gp_reg_multi {
    /**
     * Multioutput GP Regression.
     * @ref: www.gatsby.ucl.ac.uk/~snelson/thesis.pdf
     **/
    private:
      struct implementation;
      implementation* pimpl;
    public:
      /**
       * Constructor
       **/
      gp_reg_multi();
      /**
       * Destructor
       **/
      ~gp_reg_multi();
      /**
       *  Sets the multioutput kernel to be used with the multioutput
       *  regression. This kernel should have already been assigned inner
       *  kernels for each latent function.
       *  @param k : A multioutput class kernel, Default is an LMC Kernel,
       *             but you can implement your own class using the provided
       *             API.
       **/
      void set_kernel(const std::shared_ptr<multioutput_kernel_class> &k);
      /**
       *  Sets the pairs of known input and output data used to train the
       *  model.
       *  @param X : Vector of matrices, each matrix contains the inputs related
       *             to each output class.
       *  @param y : Vector of vectors, each vector contains the outputs related
       *             to each output class, and to each of the imputs.
       **/
      void set_training_set(const std::vector<arma::mat> &X,
        const std::vector<arma::vec> &y);
      /**
       *  Trains the model using the standard procedure, in accordance to the
       *  provided training set.
       *  @param max_iter : Maximum number of iterations during training.
       *  @param tol : Relative tolerance on the optimization parameters.
       **/
      double train(const int max_iter, const double tol);
      /**
       *  Trains the the model using the FITC approximation, in accordance to
       *  the provided training set.
       *  @param num_pi : Number of inducing points per output class, should be
       *                  smaller than the smallest number of inputs for any
       *                  output class, use of this parameter triggers the use
       *                  of FITC instead of the full regression.
       **/
      double train(const int max_iter, const double tol, const size_t num_pi);
      /**
       *  @param num_pi : A vector containing the number of inducing points for
       *                  each output class, each value should be smaller than
       *                  the corresponding number of inputs for the
       *                  output class, use of this parameter triggers the use
       *                  of FITC instead of the full regression.
       **/
      double train(const int max_iter, const double tol,
        const std::vector<size_t> num_pi);
       /* *
       *  @param num_pi : A vector containing the inducing points for each
       *                  output class, each matrix should be smaller than
       *                  the corresponding number of inputs for the
       *                  output class, use of this parameter triggers the use
       *                  of FITC instead of the full regression.
       * */
      double train(const int max_iter, const double tol,
        const std::vector<arma::mat> num_pi);
      /**
       *  Uses the already trained model to predict output values for new
       *  inputs provided in the parameter,this method returns the complete
       *  multivariate gaussian distribution resulting from the regression
       *  process.
       *  @param new_data : A vector of matrices containing points for which
       *                    output data is unknown in one or more of the output
       *                    classes.
       **/
      mv_gauss full_predict(const std::vector<arma::mat> &new_data);
      /**
       *  Uses the already trained model to predict output values for new
       *  inputs provided in the parameter, this method returns only the mean
       *  of the multivariate gaussian distribution resulting from the
       *  regression process, which is the "best guess" for each new input.
       *  @param new_data : A vector of matrices containing points for which
       *                    output data is unknown in one or more of the output
       *                    classes.
       **/
      arma::vec predict(const std::vector<arma::mat> &new_data) const;
      /**
       *  Returns a vector with the complete set of parameters required by the
       *  multioutput regression (pseudo-inputs if using FITC, multioutput
       *  kernel parameters and inner kernel parameters).
       **/
      std::vector<double> get_params() const;
      /**
       *  Sets all the parameters of the multioutput regression using the Vector
       *  prvided.
       *  @param params : A vector containing all parameters required by the
       *                  multioutput regression (pseudo-inputs if need be,
       *                  multioutput kernel parameters and inner kernel
       *                  parameters).
       **/
      void set_params(const std::vector<double> &params);
      enum {FULL, FITC};
    };
};

#endif
