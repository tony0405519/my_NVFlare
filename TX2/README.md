# Dockerfile for NVFlare in TX2
- This image is old nvflare with pytorch GPU, so the project we test can't work with it.

## Description
- Use `TX2/build.sh` to setup an image for nvflare.
- Use `TX2/run.sh` to run a container.

## Note
- There is no `nvidia-smi` in it, make sure the nvflare's project won't use it!
- You can find more image on: https://hub.docker.com/repositories/nctubug
