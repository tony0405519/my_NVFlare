FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.11-py3

RUN pip uninstall -y cmake
# # # RUN apt purge -y python3-yaml

RUN apt update
RUN apt -y install vim
RUN apt install -y cmake && hash -r
RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/OpenMined/TenSEAL
RUN cd TenSEAL && echo "__version__ = \"0.3.12\"" > ./tenseal/version.py && git submodule init && git submodule update && pip install .

RUN git clone https://github.com/NVIDIA/NVFlare.git
RUN cd NVFlare && git checkout main && cp -r examples/advanced/cifar10 . && cd cifar10/cifar10-real-world && pip install -r ./requirements.txt
