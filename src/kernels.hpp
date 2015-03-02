#ifndef GPLIB_KERNEL
#define GPLIB_KERNEL

namespace gplib {
  namespace kernels {
    class squared_exponential : public kernel {
      private:
        struct impementation;
        impementation *pimpl;
      public:
        squared_exponential();
        squared_exponential(const std::vector<double> &params);
        ~squared_exponential();
        arma::mat eval(const arma::mat& X, const arma::mat& Y, size_t id_out_1 = 0, size_t id_out_2 = 0);
        arma::mat derivate(size_t param_id, const arma::mat& X, const arma::mat& Y,
            size_t id_out_1 = 0, size_t id_out_2 = 0);
        size_t n_params() const;
        void setParams(const std::vector<double>& params);
        std::vector<double> get_params() const;
        std::vector<double> set_lower_bounds();
        std::vector<double> get_lower_bounds() const;
        std::vector<double> set_upper_bounds();
        std::vector<double> get_upper_bounds() const;
    };
  };
};

#endif
