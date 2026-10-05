################################################################################
# VLMotion - GPU image (RTX 5060 / CUDA 12.8)
#
# Host: driver 595+ (CUDA 13.x capable), sm_120 (Blackwell).
# ROS 2 Humble + rmw_zenoh_cpp match the 4060ti stack; Python deps are the same
# VLMotion GUI / RoboPoint worker. Source is bind-mounted at runtime.
################################################################################

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

LABEL org.opencontainers.image.title="VLMotion"
LABEL org.opencontainers.image.description="VLMotion host environment (GPU / RTX 5060)"

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
COPY docker/deps/requirements.5060.txt ./requirements.docker.txt

RUN pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.docker.txt

# colcon-core 0.21 requires setuptools<80. packaging pin matches 4060ti image.
# urchin loads the Stretch URDF for the GUI fingertip solver.
RUN python3 -m pip install --no-cache-dir \
        "setuptools>=68,<80" \
        "packaging>=24,<25" \
        "accelerate==0.30.1" \
        urchin \
        fastapi \
        uvicorn

FROM toolchain AS dev

ENV PYTHONPATH=/workspace/ros2_ws/src/vlservo:/workspace/ros2_ws/src/vlpoint
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV QT_QPA_PLATFORM=xcb
ENV NO_AT_BRIDGE=1

CMD ["/bin/bash"]
