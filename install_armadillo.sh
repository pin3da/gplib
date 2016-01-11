#!/bin/sh

wget http://sourceforge.net/projects/arma/files/armadillo-6.400.3.tar.gz
tar xf armadillo-6.400.3.tar.gz
cd armadillo-6.400.3 && ./configure && make && sudo make install && sudo ldconfig
