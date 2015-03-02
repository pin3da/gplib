#ifndef GPLIB_KERNEL
#define GPLIB_KERNEL

namespace gplib {
  namespace kernels {
    class SquaredExponential : public Kernel {
      private:
        struct Implementation;
        Implementation *pimpl;
      public:
        SquaredExponential();
        SquaredExponential(const std::vector<double> &params);
        ~SquaredExponential();
        arma::mat eval(const arma::mat& X, const arma::mat& Y, unsigned int idOut1=0, unsigned int idOut2=0);
        arma::mat derivate(unsigned int paramId, const arma::mat& X, const arma::mat& Y,
            unsigned int idOut1=0, unsigned int idOut2=0);
        unsigned int nparams() const;
        void setParams(const std::vector<double>& params);
        std::vector<double> getParams() const;
        std::vector<double> setLowerBounds();
        std::vector<double> getLowerBounds() const;
        std::vector<double> setUpperBounds();
        std::vector<double> getUpperBounds() const;
    };
  };
};

#endif
