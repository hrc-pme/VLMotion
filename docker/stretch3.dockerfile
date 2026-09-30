################################################################################
# VLMotion - CPU image (stretch3 / robot)
#
# Same base as StreamVLN's robot image: Ubuntu 22.04, ROS 2 Humble, rmw_zenoh_cpp.
# The bridge only forwards cmd_vel; cameras stay on the hellorobot driver.
################################################################################

FROM ubuntu:22.04 AS base

LABEL org.opencontainers.image.title="VLMotion"
LABEL org.opencontainers.image.description="VLMotion robot bridge (CPU / stretch3)"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ca-certificates curl git sudo gnupg2 lsb-release \
    pkg-config build-essential \
    python3 python3-dev python3-pip \
    libglib2.0-0 \
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
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-rmw-zenoh-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rosdep init 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

FROM ros AS toolchain

RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools<82" wheel \
    && pip install --no-cache-dir numpy

FROM toolchain AS dev

ENV PYTHONPATH=/workspace/ros2_ws/src/vlmotion_robot_bridge
ENV NO_AT_BRIDGE=1

CMD ["/bin/bash"]
