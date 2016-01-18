#ifndef GPLIB_H
#define GPLIB_H


/*  Version macros for compile-time API version detection                     */
#define GPLIB_VERSION_MAJOR 1
#define GPLIB_VERSION_MINOR 0
#define GPLIB_VERSION_PATCH 0

#define GPLIB_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define GPLIB_VERSION \
    GPLIB_MAKE_VERSION(GPLIB_VERSION_MAJOR, GPLIB_VERSION_MINOR, GPLIB_VERSION_PATCH)

#include "mvgauss.hpp"
#include "basic.hpp"
#include "gp.hpp"
#include "kernels.hpp"
#include "multioutput_kernels.hpp"

#endif
