#!/bin/bash

wget -q https://shoup.net/ntl/ntl-11.4.3.tar.gz
tar xzf ntl-11.4.3.tar.gz
cd ntl-11.4.3/src
./configure NTL_GMP_LIP=off
make -j8
make install
