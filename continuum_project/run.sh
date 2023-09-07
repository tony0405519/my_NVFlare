sudo docker run -it --gpus all --ipc host \
--device /dev/video0:/dev/video0 \
--device /dev/video1:/dev/video1 \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e DISPLAY=$DISPLAY \
social_dist bash
