FROM ubuntu:18.04

RUN apt-get update && apt-get -y install wget tar openssl git make cmake \
    python3 python3-pip python-pip

WORKDIR /root
ADD images ./images

ADD build.sh .
RUN ./build.sh all

ADD get-mp-spdz.sh .
RUN ./get-mp-spdz.sh

ADD *.sh *.py ./
RUN ./run.sh v1_0.25_128 images/collie.jpg ring 1 4 conv2ds cisc split
