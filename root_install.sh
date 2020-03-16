#!/usr/bin/env bash

sudo apt-get install git dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev \
libxft-dev libxext-dev
sudo apt-get install gfortran libssl-dev libpcre3-dev \
xlibmesa-glu-dev libglew1.5-dev libftgl-dev \
libmysqlclient-dev libfftw3-dev libcfitsio-dev \
graphviz-dev libavahi-compat-libdnssd-dev \
libldap2-dev python-dev libxml2-dev libkrb5-dev \
libgsl0-dev libqt4-dev
wget https://root.cern/download/root_v6.18.04.source.tar.gz
tar -xzf root_v6.18.04.source.tar.gz
mkdir root
cd root
cmake --Dall=ON ../root-6.18.04/
cmake --build . -- -j2

