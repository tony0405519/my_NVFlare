docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/aistubuntu/PoYu/tmp_dataset:/root/datasets -p 8102-8103:8102-8103 pytorch_sound:latest
