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
        arma::mat eval(const arma::mat& X, const arma::mat& Y, size_t id_out1=0, size_t id_out2=0);
        arma::mat derivate(size_t param_id, const arma::mat& X, const arma::mat& Y,
            size_t id_out1=0, size_t id_out2=0);
        size_t n_params() const;
        void set_params(const std::vector<double>& params);
        std::vector<double> get_params() const;
        std::vector<double> set_lowerBounds();
        std::vector<double> get_lowerBounds() const;
        std::vector<double> set_upperBounds();
        std::vector<double> get_upperBounds() const;
    };
  };
};

#endif
