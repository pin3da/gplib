#ifndef GPLIB_BASIC
#define GPLIB_BASIC

/* Include the basic files for this machine learning library */

#include <cmath>
#include <map>
#include <vector>

namespace gplib {
    //definition of basic constants
    const double pi = std::acos(-1);

    arma::mat upperTriangularInverse(const arma::mat& upperT);

    /* Takes a vector of real values and a boolean vector telling which dimensions are observed
     * and returns a new vector with the observed dimensions only.
     */
    arma::vec getObservedOnly(const arma::vec& vec, const std::vector<bool>& observed);

    /*
     * Splits the indices (Zero indexed) on the ones where the pradicate is true and the part it is false
     */
    void splitIndices(const std::vector<bool>& predicates, std::vector<unsigned int>& truePart, std::vector<unsigned int>& falsePart);

    /* Return true if all the values in the boolean vector are true. */
    bool allTrue(const std::vector<bool>& vec);
};

#endif
