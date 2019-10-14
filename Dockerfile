FROM ubuntu:18.04

RUN apt-get update && apt-get -y install wget tar openssl git make cmake \
    python3 python3-pip python-pip

WORKDIR /root
ADD *.sh *.py ./
ADD images ./images

RUN ./build.sh all

RUN ./get-mp-spdz.sh

RUN ./run.sh v1_0.25_128 images/collie.jpg ring 1 4
