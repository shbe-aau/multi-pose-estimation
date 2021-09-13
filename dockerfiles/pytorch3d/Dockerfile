# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

## Original source:
## https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
## Modified by shbe@create.aau.dk

ARG UBUNTU_VERSION=18.04

FROM nvidia/cudagl:10.0-devel-ubuntu${UBUNTU_VERSION} as base

# Required for Python3.6
RUN apt-get update && apt-get -q install -y \
    python3-pip \
    python3 \
    git \
    libfontconfig1-dev \
    curl

# Upgrade pip3
RUN pip3 --no-cache-dir install --upgrade \
    pip

# Install OpenCV dependencies
RUN apt-get install -y \
      libglfw3-dev \
  libglfw3

# Install cython and numpy
RUN pip3 --no-cache-dir install \
    cython \
    numpy

# Install various libraries
RUN pip3 --no-cache-dir install \
    cyglfw3 \
    imageio==2.6.0 \
    imgaug \
    progressbar \
    configparser \
    imageio \
    matplotlib \
    scikit-image \
    pypng \
    pytz \
    glumpy

# Install pytless rendering requirements
RUN pip3 --no-cache-dir install \
    pyopengl \
    pyglet \
    vispy \
    ruamel.yaml

# Install pytorch (pytorch3d depends on it...)
RUN pip3 --no-cache-dir install \
    torch==1.7.1 \
    torchvision==0.8.2 \
    torchaudio==0.7.2

# Install OpenCV
RUN pip3 --no-cache-dir install \
    opencv-python==3.4.0.14 \
    opencv-contrib-python

# Install CUB
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
RUN tar xzf 1.10.0.tar.gz
ENV CUB_HOME=$PWD/cub-1.10.0

# Install pytorch3d (forked)
ENV FORCE_CUDA=1
# RUN pip3 --no-cache-dir install 'git+https://github.com/shbe-aau/pytorch3d.git'
RUN pip3 --no-cache-dir install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0'

# Install correct version (i.e. 3.3) of libassimp and pyassimp
RUN apt-get install -y \
    libminizip1 \
    wget
RUN wget https://launchpadlibrarian.net/230132840/libassimp3v5_3.2~dfsg-3_amd64.deb
RUN apt-get install -y ./libassimp3v5_3.2~dfsg-3_amd64.deb

RUN wget https://launchpadlibrarian.net/230132835/libassimp-dev_3.2~dfsg-3_amd64.deb
RUN apt-get install -y ./libassimp-dev_3.2~dfsg-3_amd64.deb

RUN pip3 --no-cache-dir install pyassimp==3.3

# Point Python to Python3 (needed for BOP toolkit)
RUN ln -s /usr/bin/python3 /usr/bin/python
