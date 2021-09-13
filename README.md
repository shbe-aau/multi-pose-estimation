## Pose  Estimation  from  RGB  Images  of  Highly  Symmetric  Objects using  a  Novel  Multi-View  Loss  and  Differential  Rendering

## Overview

Pending...

<p align="center">
<img src='docs/overview.svg' width='800'>
<p>

## Setup

Install docker and some requirements to make the image build properly.

 - Install nvidia-docker2
 (see https://gist.github.com/Brainiarc7/a8ab5f89494d053003454efc3be2d2ef)

- Add "nvidia" as default runtime to build Docker image with CUDA support. Required for pytorch3d.
(see https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)

- Clone this repo to a folder you will share with docker, default is that the repository is a subfolder to `~/share-to-docker`. You can instead change the share variables in `dockerfile/pytorch3d/run-gpu0.sh` to point to where you place the repo if required.

- Run `bash build.sh` in `dockerfile/pytorch3d` to build the Docker image.

Tested working on Ubuntu 16.04 LTS with Docker 18.09.7 and NVIDIA docker 2.3.0, and on Ubuntu 16.04 LTS with Docker 19.03.4 and NVIDIA docker 2.3.0.

## Prepare models and data

Get background images. We use the VOC2012 dataset as background images, they can be found at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ and default is to store them in the repo under `pytorch3d/data/VOC2012`.

Get CAD files or use default as provided in `pytorch3d/data/cad-files`

Get base encoder https://dlrmax.dlr.de/get/b42e7289-7558-5da0-8f26-4c472ad830a9/ as provided from https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath

Instructions for training your own base encoder will be added later.

## Train

Start the docker container with the script in `dockerfile/pytorch3d/run-gpu0.sh`.

To train a pose regression network on top of the pretrained encoder you can run `train.py experiment_template.cfg` in the folder `pytorch3d` inside the docker container. Parameters for the training are set in the .cfg file.

## Visualize loss landscape

To visualize a loss landscape you can use the same config file as for training. Parameters under `[Sampling]` set number of points on a sphere to try, and which of those poses to treat as groundtruth (by index).

Determine poses and losses by running `plot_loss_landscape.py` in the docker container, and produce and view the final plots by running `show_loss_landscape.py`. You might want to run the second outside the docker to interact with the plots.
