
#ifndef GPLIB_GP
#define GPLIB_GP

#include <armadillo>
#include <vector>
#include <memory>

namespace gplib {

    class Kernel {
    public:
      virtual ~Kernel() = 0;
      virtual arma::mat eval(const arma::mat& X, const arma::mat& Y,
          unsigned int idOut1=0, unsigned int idOut2=0) const = 0;
      virtual arma::mat derivate(unsigned int paramId, const arma::mat& X,
          const arma::mat& Y, unsigned int idOut1=0, unsigned int idOut2=0) const = 0;
      virtual unsigned int nparams() const = 0;
      virtual void setParams(const std::vector<double>& params) = 0;
      virtual std::vector<double> getParams() const = 0;
      virtual std::vector<double> setLowerBounds()  = 0;
      virtual std::vector<double> getLowerBounds() const = 0;
      virtual std::vector<double> setUpperBounds()  = 0;
      virtual std::vector<double> getUpperBounds() const = 0;
    };

    class GPReg {
    private:
      struct Implementation;
      Implementation* pimpl;
    public:
      GPReg();
      ~GPReg();
      void setKernel(const std::shared_ptr<Kernel>& k);
      std::shared_ptr<Kernel> getKernel() const;
      void setTrainingSet(const arma::mat &X, const arma::vec& y);
      void train();
      MVGauss fullPredict(const arma::mat& newData) const;
      arma::vec predict(const arma::mat& newData) const;
    };
};

#endif
