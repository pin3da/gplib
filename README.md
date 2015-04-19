gplib
=====

Build status : [![Build Status](https://travis-ci.org/pin3da/gplib.svg?branch=master)](https://travis-ci.org/pin3da/gplib)

C++ Gaussian Process Library

Requirements/Dependencies
------------

- [Armadillo](http://arma.sourceforge.net/)
- C++11 compliant compiler. (g++ >= 4.7, clang++ >= 3.5)
- (Optinal, only for test) boost test suites.
- NLOpt Optimization library.

Example in debian-based distributions

    $apt_pref update
    $apt_pref install libarmadillo-dev g++ libboost-test-dev libnlopt-dev


Note: Install Armadillo from the source code, for a more updated
version. [link](http://arma.sourceforge.net/download.html)

Installation
------------

Installation can be done by the standard make && make install. If the boost
unittest framework is install check and installcheck can be run for sanity
checking.

    git clone https://github.com/pin3da/gplib.git
    cd gplib
    make
    make check
    make install
    make installcheck

The most commonly useful overrides are setting CXX, to change the compiler
used, and PREFIX to change install location. The CXX prefix should be used on
all targets as the compiler version is used in the build path. PREFIX is only
relevant for the install target.

Example compiling with clang++

    export CXX=clang++
    make
    make check


Contributing
============

Contribution to this library is welcome and it is suggested using pull requests
in github that will then be reviewed and merged or commented on. A more specific
contribution guideline is outlined on the [zmq site](http://zeromq.org/docs:contributing)

Please feel free to add yourself to the [AUTHORS](https://github.com/pin3da/gplib/blob/master/AUTHORS) file in an alphanumerically
sorted way before you raise the pull request.

Documentation
=============

Most of the code is now commented with doxygen style tags, and a basic configuration file to generate them is in the root directory.

To build the documentation with doxygen use

    doxygen

And the resulting html or latex docs will be in the docs/html or docs/latex directories.

An online version of this documentation brought to you by [gh-pages](http://pin3da.github.io/gplib/)

*Disclaimer : This version may be outdated*

Licensing
=========

The library is released under the MPLv2 license.

Please see LICENSE for full details.

_______

Developed by In-silico.
