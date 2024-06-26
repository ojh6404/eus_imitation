FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install essential packages
RUN apt update && apt install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    curl \
    wget \
    build-essential \
    sudo \
    git \
    lsb-release \
    vim \
    tmux \
    gcc \
    g++ \
    make \
    libtool \
    texinfo \
    dpkg-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-distutils python3.11-venv python3.11-dev python3-pip python-is-python3\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install -U pycryptodomex rospkg
RUN sudo pip install -U pip setuptools cython numpy setuptools_scm opencv-python psutil wrapt

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN \
  useradd user && \
  echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
  chmod 0440 /etc/sudoers.d/user && \
  mkdir -p /home/user && \
  chown user:user /home/user && \
  chsh -s /bin/bash user

RUN echo 'root:root' | chpasswd
RUN echo 'user:user' | chpasswd
# setup sources.list
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# setup keys
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

# install ros core
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-rosdep \
    python3-rosinstall \
    python3-osrf-pycommon \
    python3-wstool \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt update && apt install -y --no-install-recommends \
    ros-noetic-image-transport-plugins \
    ros-noetic-jsk-tools \
    ros-noetic-jsk-common \
    ros-noetic-jsk-topic-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user
USER user
SHELL ["/bin/bash", "-c"]


ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA=True
ENV CUDA_HOME=/usr/local/cuda/
# trick for cuda because cuda is not available when building docker image
ENV FORCE_CUDA="1" TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# to use gdown, you need to add path to ~/.local/bin
ENV PATH="$PATH:$HOME/.local/bin"

########################################
########### WORKSPACE BUILD ############
########################################
# Installing catkin package
RUN mkdir -p ~/catkin_ws/src
RUN sudo pip install git+https://github.com/catkin/catkin_tools.git
RUN sudo rosdep init && rosdep update && sudo apt update
RUN git clone https://github.com/ros-o/ros_comm.git -b obese-devel ~/catkin_ws/src/ros_comm # for python3.11 compatibility
RUN git clone https://github.com/ojh6404/eus_imitation.git ~/catkin_ws/src/eus_imitation
RUN cd ~/catkin_ws/src/ &&\
    source /opt/ros/noetic/setup.bash &&\
    rosdep install --from-paths . -i -r -y
RUN source /opt/ros/noetic/setup.bash &&\
    cd ~/catkin_ws/src/eus_imitation/imitator_ros && ./prepare.sh &&\
    cd ~/catkin_ws && catkin init && catkin build &&\
    rm -rf ~/.cache/pip

# # to avoid conflcit when mounting
# RUN rm -rf ~/catkin_ws/src/eus_imitation/eus_imitation_core/launch
# RUN rm -rf ~/catkin_ws/src/eus_imitation/eus_imitation_core/node_scripts
# RUN rm -rf ~/catkin_ws/src/eus_imitation/eus_imitation_core/scripts

########################################
########### ENV VARIABLE STUFF #########
########################################
RUN touch ~/.bashrc
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

CMD ["bash"]
