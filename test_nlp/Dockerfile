FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

RUN apt -m update
RUN apt -y install vim
RUN apt -y install libssl-dev
RUN apt -y install git
RUN apt -y install make
RUN apt -y install --reinstall ca-certificates
RUN apt -y install software-properties-common

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt -y install gcc-9
RUN apt -y install g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN cd /usr/bin && ln -sfn /usr/bin/gcc-9 aarch64-linux-gnu-gcc && ln -sfn /usr/bin/g++-9 aarch64-linux-gnu-g++

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools

RUN git clone https://github.com/OpenMined/TenSEAL
RUN cd TenSEAL && echo "__version__ = \"0.3.0\"" > ./tenseal/version.py && echo "string(APPEND CMAKE_CXX_FLAGS \" -latomic\")" >> CMakeLists.txt && echo "string(APPEND CMAKE_C_FLAGS \" -latomic\")" >> CMakeLists.txt && git submodule init && git submodule update && pip3 install .

RUN git clone https://github.com/NVIDIA/NVFlare.git
RUN cd NVFlare && git checkout main && cp -r examples/advanced/nlp-ner . && cd nlp-ner &&  pip3 install -r ./requirements.txt
