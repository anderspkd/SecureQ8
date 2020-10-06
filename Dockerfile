FROM ubuntu:18.04

RUN apt-get update && apt-get -y install wget tar openssl git make cmake \
    python3 python3-pip python-pip clang libsodium-dev autoconf automake \
    libtool yasm texinfo libboost-dev libssl-dev libboost-system-dev \
    libboost-thread-dev libgmp-dev rsync ssh openssh-server procps

WORKDIR /root
ADD images ./images

ADD build.sh .
RUN ./build.sh all

RUN git clone https://github.com/data61/MP-SPDZ -b v0.1.8

ADD build-ntl.sh .
RUN ./build-ntl.sh

ADD build-mp-spdz.sh .
RUN ./build-mp-spdz.sh

ADD ssh_config .ssh/config
ADD setup-ssh.sh .
RUN ./setup-ssh.sh

ADD *.sh *.py HOSTS ./
RUN service ssh start; ./run-remote.sh v1_0.25_128 images/collie.jpg ring prob 4
