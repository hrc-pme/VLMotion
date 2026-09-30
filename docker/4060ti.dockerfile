################################################################################
# VLMotion - GPU image (4060 Ti / CUDA 12.1)
#
# Hardware stack matches StreamVLN (CUDA 12.1, ROS 2 Humble, rmw_zenoh_cpp).
# Python deps are the VLMotion GUI / RoboPoint worker, not StreamVLN.
# Source is bind-mounted at runtime.
################################################################################

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

LABEL org.opencontainers.image.title="VLMotion"
LABEL org.opencontainers.image.description="VLMotion host environment (GPU / 4060ti)"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ca-certificates curl wget git sudo gnupg2 lsb-release \
    software-properties-common pkg-config build-essential \
    python3 python3-dev python3-venv python3-pip \
    dialog whiptail \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libx11-xcb1 libxcb1 libxcb-xinerama0 libxkbcommon-x11-0 \
    python3-pyqt5 \
    ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

FROM base AS ros

ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -q -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-cv-bridge \
    ros-humble-rmw-zenoh-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rosdep init 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

FROM ros AS toolchain

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir "setuptools<82" wheel

WORKDIR /opt/vlmotion-deps
COPY docker/deps/requirements.4060ti.txt ./requirements.docker.txt

RUN pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.docker.txt

FROM toolchain AS dev

ENV PYTHONPATH=/workspace/ros2_ws/src/vlservo:/workspace/ros2_ws/src/vlpoint
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV QT_QPA_PLATFORM=xcb
ENV NO_AT_BRIDGE=1

CMD ["/bin/bash"]
