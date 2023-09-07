ARG CUDA_VERSION=11.4.3
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.8
ARG OPENCV_VERSION=4.2.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update && \
    apt-get -qq install  \
#   python :
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        libpython${PYTHON_VERSION} \
        libpython${PYTHON_VERSION}-dev \
        python-dev \
        python3-setuptools \
#   developement tools, opencv image/video/GUI dependencies, optimiztion packages , etc ...  :
        apt-utils \
        autoconf \
        automake \
        checkinstall \
        cmake \
        gfortran \
        git \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libeigen3-dev \
        libexpat1-dev \
        libglew-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libpostproc-dev \
        libpq-dev \
        libqt5opengl5-dev \
        libsm6 \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libtiff-dev \
        libtool \
        libv4l-dev \
        libwebp-dev \
        libxext6 \
        libxrender1 \
        libxvidcore-dev \
        pkg-config \
        protobuf-compiler \
        qt5-default \
        unzip \
        wget \
        yasm \
        zlib1g-dev \
        vim \
#   GStreamer :
        libgstreamer1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-doc \
        gstreamer1.0-tools \
        gstreamer1.0-x \
        gstreamer1.0-alsa \
        gstreamer1.0-gl \
        gstreamer1.0-gtk3 \
        gstreamer1.0-qt5 \
        gstreamer1.0-pulseaudio \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge   --auto-remove && \
    apt-get clean

# install new pyhton system wide :
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --config python3

# numpy for the newly installed python :
RUN wget https://bootstrap.pypa.io/get-pip.py  && \
    python${PYTHON_VERSION} get-pip.py --no-setuptools --no-wheel && \
    rm get-pip.py && \
    pip install numpy

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip

WORKDIR opencv-${OPENCV_VERSION}/
RUN mkdir build
WORKDIR build/

# -DCUDA_nppicom_LIBRARY=stdc++ is duplicate in CUDA 11 but opencv 4.2 not fix yet
# CUDNN buggy
ENV python_exec='/usr/bin/python3'
RUN export include_dir=`${python_exec} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"` && \
    export library=`${python_exec} -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"` && \
    export site_packages=`${python_exec} -c 'import site; print(site.getsitepackages()[0])'` && \
    cmake -DWITH_CUDA=ON -DWITH_CUBLAS=1 -DWITH_TBB=ON -DWITH_GSTREAMER=ON -DWITH_LIBV4L=ON -DWITH_FFMPEG=ON -DWITH_V4L=ON -DWITH_QT=OFF -DWITH_OPENGL=ON -DWITH_OPENMP=ON -DWITH_EIGEN=ON -DWITH_JPEG=ON    -DWITH_NVCUVID=ON -DWITH_NVCUVENC=ON  -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DCUDA_ARCH_BIN="7.5" -DCUDA_ARCH_PTX="" -DCUDA_nppicom_LIBRARY=stdc++  -DOPENCV_DNN_CUDA=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules/ -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_python2=ON -DBUILD_opencv_python3=ON -DBUILD_opencv_cudacodec=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=ON -DBUILD_JPEG=ON -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON3_EXECUTABLE=${python_exec} -DPYTHON_DEFAULT_EXECUTABLE=${python_exec} -DPYTHON3_INCLUDE_DIR=${include_dir} -DPYTHON3_LIBRARY=${library} -DPYTHON3_PACKAGES_PATH=${site_packages} ..

RUN make -j16 && make install

RUN pip install argparse==1.4.0

ENV NVIDIA_DRIVER_CAPABILITIES all
ENV XDG_RUNTIME_DIR "/tmp"

WORKDIR /
RUN git clone https://github.com/OhMyBuggg/Social-Distancing-AI.git
WORKDIR Social-Distancing-AI/
RUN wget https://pjreddie.com/media/files/yolov3.weights && \
    mv yolov3.weights ./models/
