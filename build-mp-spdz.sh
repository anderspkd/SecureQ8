#!/bin/bash

cd MP-SPDZ
echo CXX = clang++ >> CONFIG.mine
echo USE_NTL = 1 >> CONFIG.mine
make -j8 tldr
mkdir static
make -j8 {static/,}{{replicated,ps-rep}-{ring,field},semi2k,hemi,cowgear,spdz2k}-party.x
