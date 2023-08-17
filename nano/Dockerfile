FROM nvcr.io/nvidia/l4t-base:r32.7.1

RUN apt update
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

RUN apt -y install python3.8
RUN apt -y install libpython3.8-dev
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --set python /usr/bin/python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --set python3 /usr/bin/python3.8
RUN apt install python3-pip -y
RUN python -m pip install --upgrade pip
RUN python -m pip install -U setuptools

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.1/cmake-3.27.1.tar.gz && tar -zxvf cmake-3.27.1.tar.gz && rm cmake-3.27.1.tar.gz
RUN cd cmake-3.27.1 && ./bootstrap -- -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" && make && make install && hash -r
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN cd /usr/bin && ln -sfn /usr/bin/gcc-9 aarch64-linux-gnu-gcc && ln -sfn /usr/bin/g++-9 aarch64-linux-gnu-g++

# exit ?

RUN git clone https://github.com/OpenMined/TenSEAL
RUN cd TenSEAL && echo "__version__ = \"0.3.12\"" > ./tenseal/version.py && echo "string(APPEND CMAKE_CXX_FLAGS \" -latomic\")" >> CMakeLists.txt && echo "string(APPEND CMAKE_C_FLAGS \" -latomic\")" >> CMakeLists.txt && git submodule init && git submodule update && pip install .

RUN git clone https://github.com/NVIDIA/NVFlare.git
RUN cd NVFlare && git checkout main && cp -r examples/advanced/cifar10 . && cd cifar10/cifar10-real-world && pip install -r ./requirements.txt
