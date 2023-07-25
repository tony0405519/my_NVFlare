sudo docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./my-workspace/:/my-workspace -p 8887:8888 nvflare-pt:latest
