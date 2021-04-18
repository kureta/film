docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidiactl -p 8888:8888 -v  "`pwd`:/src" zak-ddsp
