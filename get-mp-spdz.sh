#!/bin/bash

wget -O - https://github.com/data61/MP-SPDZ/releases/download/v0.1.8/mp-spdz-0.1.8.tar.xz | tar xJ
rmdir MP-SPDZ
mv mp-spdz-0.1.8 MP-SPDZ
cd MP-SPDZ
Scripts/tldr.sh
