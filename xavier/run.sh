sudo docker run -it --runtime=nvidia --ipc=host -e DISPLAY=$DISPLAY -p 8000-9000:8000-9000 --ulimit memlock=-1 --ulimit stack=67108864 nvflare_cifar_xavier
