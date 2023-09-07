# EdgeDeviceDocker
This is a docker and tutorial for edge device.
- ubuntu 20.04 PC
    - Docker version 24.0.1
- Nvidia Jetson AGX Xavier: Ubuntu 20.04
- Jetson Nano 2GB
    - Too low to use cpu.
    - Use `LXDE-based desktop` to get more RAM.
    - It doesn't have enough RAM to build my docker.

## Steps
1. run `build.sh` to build docker image.
2. run `$ xhost +` in local to allow host connection.
3. run `run.sh` to create and start container.

## Caution
- Make sure `$DISPLAY` is the same in local and docker. Run `$ echo $DISPLAY` to check. If not, set docker's variable to right. Run `$ export DISPAY={what's in local}`.
- The package version list can't be the same in PC and Xavier, I'm still trying to figure out the problem.
- `Dockerfile_xavier is also compatible with PC`
- Jetson Nano has just one device, maybe it's due to the ubuntu version.
    - https://unix.stackexchange.com/questions/512759/multiple-dev-video-for-one-physical-device

## Update
- There is a built docker.
    - plugin-motion-detector
        - everything goes well, just run it.
            - Xavier need to use --runtime=nvidia
        - Can not show display.
    - plugin-objectcounter
        - default use waggle, need to change to local device. 
        - at least 8GB remain space -> `waggle/plugin-base:1.1.1-ml-torch1.9` this image's file structure isn't same as `waggle/plugin-base:1.1.1-base`, COPY local in it now.(TBD)
        - nano will crash.
        - CUDA different between PC and edge device (11 vs 10.2) -> update docker didn't work
    - sound-event-detection
        - everything goes well, just run it.
            - Xavier need to use --runtime=nvidia
        - Can not show display.

## Note
- `$ docker builder prune` to delete docker build cache.


## Result table
| porject | arch. | platform | status | comments |
| ------- | -------- | ------ | ------ | -------- |
| motion detector | amd64 | Ubuntu host | ok | No gpu used in code |
|                 | arm64 | Jetson Nano | ok | (No gpu used in code) 1. use --runtime=nvidia instead of --gpus all 2. can't show display |
|                 | arm64 | Jetson Xavier | ok | (No gpu used in code) 1. use --runtime=nvidia instead of --gpus all 2. can't show display |
|                 | arm64 | Jetson TX2 | ok | (No gpu used in code) 1. use --runtime=nvidia instead of --gpus all 2. can't show display |
| object counter  | amd64 | Ubuntu host | ok | CUDA 11.4, different image between amd64 and arm64 |
|                 | arm64 | Jetson Nano | failed | CUDA version 10.2 failed to run code, use newer image(CUDA 11.4) will cause another error from opencv |
|                 | arm64 | Jetson Xavier | failed | CUDA version 10.2 failed to run code, use newer image(CUDA 11.4) will cause another error from opencv |
|                 | arm64 | Jetson TX2 | failed | CUDA version 10.2 failed to run code, use newer image(CUDA 11.4) will cause another error from opencv |
| sound detector  | amd64 | Ubuntu host | ok | No cuda used |
|                 | arm64 | Jetson Nano | ok | No cuda used |
|                 | arm64 | Jetson Xavier | ok | No cuda used |
|                 | arm64 | Jetson TX2 | ok | No cuda used |
