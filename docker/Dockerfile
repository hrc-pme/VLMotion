FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=humble \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    RMW_IMPLEMENTATION=rmw_zenoh_cpp

SHELL ["/bin/bash", "-c"]

# This image is an application client only. The robot driver, RealSense
# drivers, calibration data, audio stack, and device access live in
# helloRobot_stretch3.
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales tzdata xauth netcat-openbsd \
        python3-colcon-common-extensions \
        python3-pip \
        python3-numpy python3-opencv python3-pil python3-pyqt5 \
        python3-requests python3-sklearn python3-yaml \
        ros-humble-rmw-zenoh-cpp \
        ros-humble-audio-common-msgs \
        ros-humble-cv-bridge \
        ros-humble-control-msgs \
        ros-humble-geometry-msgs \
        ros-humble-nav-msgs \
        ros-humble-rosidl-default-generators \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs \
        ros-humble-tf2-geometry-msgs \
        ros-humble-tf2-ros \
        ros-humble-trajectory-msgs \
        ros-humble-visualization-msgs \
    && sed -i 's/^# *\(en_US\.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        "numpy>=1.24,<1.25" \
        "faster-whisper>=1.1.0,<2" \
        "opencc-python-reimplemented>=0.1.7,<1"

WORKDIR /workspace/ros2_ws

# Keep a runnable copy in the image. Compose mounts the host sources at
# runtime and rebuilds these same packages with --symlink-install.
COPY ros2_ws/src/audio_common/speech_recognition_msgs src/speech_recognition_msgs
COPY ros2_ws/src/vlmotion_stt src/vlmotion_stt
COPY ros2_ws/src/vlpoint src/vlpoint
COPY ros2_ws/src/white_point_pipeline src/white_point_pipeline

RUN source /opt/ros/humble/setup.bash \
    && colcon build \
        --symlink-install \
        --packages-select speech_recognition_msgs vlmotion_stt vlpoint white_point_pipeline \
        --cmake-args -DBUILD_TESTING=OFF

COPY docker/start-vlmotion.sh /usr/local/bin/start-vlmotion
RUN chmod +x /usr/local/bin/start-vlmotion \
    && mkdir -p /workspace/Outputs /tmp/runtime-root \
    && chmod 700 /tmp/runtime-root

WORKDIR /workspace
CMD ["/usr/local/bin/start-vlmotion"]
