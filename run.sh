#docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/nvidia0 -p 8888:8888 -v  "`pwd`:/src" zak-ddsp
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidiactl -p 8888:8888 -v  "`pwd`:/src" zak-ddsp
