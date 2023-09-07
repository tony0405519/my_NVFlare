sudo docker run -it --ipc host \
--device /dev/video0:/dev/video0 \
--device /dev/video1:/dev/video1 \
--device /dev/snd:/dev/snd \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e DISPLAY=$DISPLAY \
-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
-v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
-v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
--group-add $(getent group audio | cut -d: -f3) \
sound_detect bash
