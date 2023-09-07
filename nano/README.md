# Dockerfile for NVFlare in Nano
- This image is new nvflare without pytorch GPU, so it can't train.

## Description
- Use `nano/build.sh` to setup an image for nvflare.
- Use `nano/run.sh` to run a container.

## Note
- There is no `nvidia-smi` in it, make sure the nvflare's project won't use it!
- You can find more image on: https://hub.docker.com/repositories/nctubug

## Build on Xavier
* (前面有cache沒紀錄到）
* Build Tenseal: 1429.6s
* Install NVFlare & requirement.txt: 207.4s
