
#ifndef GPLIB_GP
#define GPLIB_GP

#include <armadillo>
#include <vector>
#include <memory>

namespace gplib {

    class kernel_class {
    public:
      virtual ~kernel_class() = 0;
      virtual arma::mat eval(const arma::mat& X, const arma::mat& Y,
          size_t id_out_1=0, size_t id_out_2=0) const = 0;
      virtual arma::mat derivate(size_t param_id, const arma::mat& X,
          const arma::mat& Y, size_t id_out_1=0, size_t id_out_2=0) const = 0;
      virtual size_t n_params() const = 0;
      virtual void set_params(const std::vector<double>& params) = 0;
      virtual std::vector<double> get_params() const = 0;
      virtual std::vector<double> set_lower_bounds()  = 0;
      virtual std::vector<double> get_lower_bounds() const = 0;
      virtual std::vector<double> set_upper_bounds()  = 0;
      virtual std::vector<double> get_upper_bounds() const = 0;
    };

    class gp_reg {
    private:
      struct implementation;
      implementation* pimpl;
    public:
      gp_reg();
      ~gp_reg();
      void set_kernel(const std::shared_ptr<kernel_class>& k);
      std::shared_ptr<kernel_class> get_kernel() const;
      void set_training_set(const arma::mat &X, const arma::vec& y);
      void train();
      mv_gauss full_predict(const arma::mat& new_data) const;
      arma::vec predict(const arma::mat& new_data) const;
    };
};

#endif
