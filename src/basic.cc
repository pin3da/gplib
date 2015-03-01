
#include "gplib.hpp"

using namespace arma;
using namespace std;

namespace gplib {

    mat upperTriangularInverse(const mat& upperT) {
        unsigned int D = upperT.n_rows; myassert(D == upperT.n_cols);
        mat ans(D,D);
        ans.fill(0.0);
        vector<double> tmp(D);
        for (unsigned int i=0; i<D; i++) {
            ans(i,i) = 1.0/upperT(i,i);
            for (unsigned int j=i+1; j<D; j++) tmp[j] = upperT(i,j)/upperT(i,i);
            for (unsigned int j=i+1; j<D; j++) {
                double factor = ans(i,j) = -tmp[j] / upperT(j,j);
                for (unsigned int k=j+1; k<D; k++) {
                    tmp[k] += factor*upperT(j,k);
                }
            }
        }
        return ans;
    }

    arma::vec getObservedOnly(const arma::vec& vec, const vector<bool>& observed) {
        myassert(vec.n_elem == observed.size());
        vector<double> tmp;
        for (unsigned int i = 0; i < observed.size(); i++)
            if (observed[i])
                tmp.push_back(vec[i]);
        return arma::vec(tmp);
    }

    void split_indices(const vector<bool>& predicates, vector<unsigned int>& truePart, vector<unsigned int>& falsePart) {
        for (unsigned int i=0; i<predicates.size(); i++) {
            if (predicates[i]) truePart.push_back(i);
            else falsePart.push_back(i);
        }
    }

    bool allTrue(const vector<bool>& vec) {
        for (unsigned int i=0; i<vec.size(); i++) if(!vec[i]) return false;
        return true;
    }

};
