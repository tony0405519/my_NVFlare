sudo docker run -it --runtime=nvidia --ipc=host -e DISPLAY=$DISPLAY -p 8102-8103:8102-8103 --ulimit memlock=-1 --ulimit stack=67108864 nvflare_cifar_tx2
