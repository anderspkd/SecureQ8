#!/bin/bash

wget -O - https://github.com/data61/MP-SPDZ/releases/download/v0.1.2/mp-spdz-0.1.2.tar.xz | tar xJ
rmdir MP-SPDZ
mv mp-spdz-0.1.2 MP-SPDZ
cd MP-SPDZ
Scripts/tldr.sh
