FROM python:2.7.13
FROM ubuntu:16.04
FROM gcr.io/tensorflow/tensorflow:0.11.0rc2
FROM zeromq/zeromq

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        && \
    python -m ipykernel.kernelspec

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install \
http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl

RUN apt-get update && apt-get install -y \
    php5-mcrypt \
    python-pip
RUN pip install python
RUN pip install tflearn
RUN pip install pyyaml
#GRL config

RUN apt-get update && apt-get install -y g++ cmake git subversion

RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libgl1-mesa-dev \
    freeglut3-dev \
    libz-dev \
    libqt4-opengl-dev \
    libtinyxml-dev \
    libmuparser-dev \
    libode-dev \
    liblua5.1-dev \
    libprotoc-dev \
    protobuf-compiler 

RUN git clone https://github.com/divyamrast/grl.git -b master; \
    cd grl; \
    mkdir build; \
    cd build; \
    cmake -DCMAKE_BUILD_TYPE=Debug ..; \
    make -j5


