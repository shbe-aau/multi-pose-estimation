# Setup

 - Install nvidia-docker2
 (see https://gist.github.com/Brainiarc7/a8ab5f89494d053003454efc3be2d2ef)

- Add "nvidia" as default runtime to build Docker image with CUDA support. Required for pytorch3d.
(see https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)

- Run `bash build.sh` to build the Docker image.

Tested working on Ubuntu 16.04 LTS with Docker 18.09.7 and NVIDIA docker 2.3.0, and on Ubuntu 16.04 LTS with Docker 19.03.4 and NVIDIA docker 2.3.0.
