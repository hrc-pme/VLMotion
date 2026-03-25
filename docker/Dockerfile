FROM ubuntu:22.04 AS base

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble

# GUI / Qt support using X11
ENV QT_QPA_PLATFORM=xcb
ENV QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]

# Set timezone and install basic utilities with GUI dependencies
RUN echo 'Asia/Taipei' > /etc/timezone \
    && ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # Basic system utilities
        tzdata dirmngr gnupg2 lsb-release locales sudo \
        build-essential software-properties-common \
        git wget unzip curl zip iputils-ping udev vim \
        # GUI and X11 support
        x11-apps xauth dbus-x11 xterm \
        libasound2-dev ffmpeg libsm6 libxext6 libxrender1 \
        libx11-xcb1 libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 \
        libxcb-xinerama0 libxcb-randr0 libxcb-image0 libxcb-keysyms1 libxcb-icccm4 \
        libxcb-cursor0 libxkbcommon-x11-0 libgl1-mesa-glx libglu1-mesa mesa-utils \
        libqt5gui5 libqt5widgets5 libqt5x11extras5 \
        # Audio support
        portaudio19-dev python3-dev \
        # Python basics
        python3-pip python3-setuptools python3-wheel \
    && sed -i 's/^# *\(en_US\.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers


################################################################################################
# Stretch Robot stage
# - Installs Stretch robot-specific software and dependencies
################################################################################################
FROM base AS stretch-robot

# Switch to root for system installations
USER root
ENV HOME=/root

# Stretch environment variables
ARG HELLO_FLEET_ID=stretch-se3-3092
ENV HELLO_FLEET_ID=$HELLO_FLEET_ID
ENV HELLO_FLEET_PATH=$HOME/stretch_user

# Clone and setup Stretch installation
RUN git clone https://github.com/hello-robot/stretch_install.git $HOME/stretch_install \
    && cd $HOME/stretch_install \
    && git checkout cb4ae97e1c31bc86bafaa0887131093f357b535f \
    && mkdir -p $HOME/stretch_user/$HELLO_FLEET_ID

# Setup Stretch configuration
RUN cd $HOME/stretch_install \
    && echo "HELLO_FLEET_ID=$HELLO_FLEET_ID" >> hello-robot.conf \
    && mkdir /etc/hello-robot \
    && mv hello-robot.conf /etc/hello-robot \
    && cp factory/22.04/stretch_about.png /etc/hello-robot/ \
    && mkdir -p $HOME/.local/bin $HOME/repos $HOME/stretch_user \
    && mkdir -p $HOME/stretch_user/{log,debug,maps,models} \
    && cd factory/22.04 \
    && cp hello_robot_lrf_off.py /usr/bin/ \
    && cp hello_robot_pimu_ping.py /usr/bin/ \
    && cp hello_robot_pimu_ping.sh /usr/bin/ \
    && rm -rf $HOME/stretch_install

# Install Stretch system dependencies
RUN apt-get update && \
    apt-get install -y \
        # Core dependencies
        apt-transport-https ca-certificates cython3 ipython3 \
        libnss3-tools liburdfdom-tools net-tools python3-serial \
        ubuntu-sounds unzip screen \
        # GStreamer support
        gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0 \
        gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly gstreamer1.0-tools \
        libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
        # Audio and speech
        festival festvox-kallpc16k \
        # Python packages
        python3-autobahn python3-bson python3-pyaudio python3-tornado \
        python3-tqdm python3-transforms3d python3-twisted python3-seaborn \
        python3-apt python3-pcl python3-pykdl python3-gi python3-pil \
        python3-requests python3-pytest python3-numpy python3-numpy-dev \
    && rm -rf /var/lib/apt/lists/*


################################################################################################
# ROS2 Humble stage
# - Installs ROS2 Humble and development tools
################################################################################################
FROM stretch-robot AS ros2-humble

# Setup ROS2 repository
RUN set -eux; \
    key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
    mkdir -p /usr/share/keyrings; \
    gpg --batch --export "$key" > /usr/share/keyrings/ros2-latest-archive-keyring.gpg; \
    gpgconf --kill all; \
    rm -rf "$GNUPGHOME"

RUN echo "deb [signed-by=/usr/share/keyrings/ros2-latest-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list

# Note: Skipping Intel RealSense repository due to GPG key verification issues
# librealsense2 packages will be installed from default repos if available

# Install ROS2 Humble desktop-full and additional packages
RUN apt-get update && \
    apt-get install -y \
        # ROS2 core
        ros-humble-desktop-full \
        ros-humble-ament-cmake ros-humble-ament-cmake-python ros-humble-ament-cmake-ros \
        ros-humble-rosidl-default-generators ros-humble-rosidl-default-runtime \
        ros-humble-rosidl-generator-py \
        # Development tools
        python3-colcon-common-extensions python3-colcon-mixin python3-colcon-clean \
        python3-rosdep python3-vcstool python3-argcomplete ros-dev-tools cmake \
        # Workspace-specific ROS packages
        ros-humble-rclpy ros-humble-rclcpp ros-humble-rclcpp-components \
        ros-humble-std-msgs ros-humble-std-srvs ros-humble-geometry-msgs \
        ros-humble-sensor-msgs ros-humble-nav-msgs ros-humble-control-msgs \
        ros-humble-trajectory-msgs ros-humble-actionlib-msgs ros-humble-builtin-interfaces \
        ros-humble-diagnostic-msgs ros-humble-diagnostic-updater ros-humble-diagnostic-aggregator \
        ros-humble-visualization-msgs ros-humble-tf2 ros-humble-tf2-ros ros-humble-tf2-eigen \
        ros-humble-tf2-geometry-msgs ros-humble-launch-ros ros-humble-launch-xml \
        ros-humble-angles ros-humble-message-filters ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher-gui ros-humble-rviz2 \
        # Navigation
        ros-humble-nav2-amcl ros-humble-nav2-bringup ros-humble-nav2-map-server \
        ros-humble-nav2-msgs ros-humble-slam-toolbox \
        ros-humble-teleop-twist-keyboard ros-humble-teleop-twist-joy ros-humble-joy \
        # RealSense (ROS wrapper, librealsense2 system packages installed via pyrealsense2 pip)
        ros-humble-realsense2-camera ros-humble-realsense2-camera-msgs \
        ros-humble-realsense2-description \
        # Image processing
        ros-humble-cv-bridge ros-humble-image-transport \
        # Point cloud
        ros-humble-pcl-conversions ros-humble-pcl-ros \
        ros-humble-octomap ros-humble-octomap-msgs ros-humble-octomap-ros \
        # LIDAR
        ros-humble-laser-filters ros-humble-rplidar-ros \
        # ROS bridge
        ros-humble-rosbridge-suite \
        # Utilities
        ros-humble-sensor-msgs-py ros-humble-tf-transformations \
        ros-humble-urdfdom-py ros-humble-xacro \
        # Testing
        ros-humble-launch-pytest ros-humble-ament-cmake-pytest \
        ros-humble-launch-testing ros-humble-launch-testing-ros \
        ros-humble-launch-testing-ament-cmake ros-humble-ament-lint-auto \
        ros-humble-ament-lint-common ros-humble-ament-flake8 \
        ros-humble-ament-pep257 ros-humble-ament-copyright \
        # System dependencies (librealsense2 system libs removed - use pyrealsense2 pip package instead)
        libeigen3-dev libpcl-dev libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep and setup colcon
RUN rosdep init || true \
    && rosdep update --rosdistro $ROS_DISTRO \
    && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
    && colcon mixin update \
    && colcon metadata add default https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml \
    && colcon metadata update


################################################################################################
# Python Dependencies stage
# - Sets up Python environment with proper version constraints
################################################################################################
FROM ros2-humble AS python-deps

# Global pip constraints to prevent version conflicts (without setuptools constraint)
RUN printf "numpy<2\n" > /etc/pip-constraints.txt \
    && mkdir -p /etc/pip \
    && printf "[global]\nconstraint = /etc/pip-constraints.txt\n" > /etc/pip/pip.conf
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Install Python packages with proper version control
# Pin pip to <25 to avoid editable install issues with colcon/ament
RUN python3 -m pip install --no-cache-dir "pip==24.2" \
    && python3 -m pip install --no-cache-dir "wheel<1" \
    && python3 -m pip install --no-cache-dir "packaging>=24.0" \
    && python3 -m pip install --no-cache-dir --ignore-installed "sympy>=1.13.3" \
    && python3 -m pip install --no-cache-dir --ignore-installed "numpy==1.24.2" "scipy==1.11.4" \
    && python3 -m pip install --no-cache-dir --ignore-installed "numba==0.60.0" "llvmlite==0.43.0"

# Ensure setuptools supports PEP 660 editable installs (for --symlink-install)
# Match version that worked in your test environment
RUN python3 -m pip install --no-cache-dir "setuptools>=68,<75"

# Preinstall build helper; avoid setuptools-scm to reduce noise in builds
RUN python3 -m pip install --no-cache-dir build

# Install Stretch Python packages (enable build isolation for PEP 517 backends)
RUN PIP_CONSTRAINT="" python3 -m pip install --no-cache-dir --no-warn-script-location --ignore-installed \
        transforms3d \
        hello-robot-stretch-body \
        hello-robot-stretch-body-tools \
        hello-robot-stretch-factory \
        hello-robot-stretch-tool-share \
        hello-robot-stretch-diagnostics \
        hello-robot-stretch-urdf \
        prompt_toolkit

# Install additional Python packages
RUN python3 -m pip install --no-cache-dir \
        # Audio and ML
        soundfile transformers accelerate sentencepiece protobuf pyaudio==0.2.14 \
        # Computer vision
        opencv-contrib-python==4.8.1.78 pyrealsense2 \
        # Utilities
        transforms3d Cython build \
        --ignore-installed blinker

# Install PyTorch with CPU-only version
RUN python3 -m pip install --no-cache-dir \
        torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Clean up OpenCV Qt plugins that might conflict
RUN OPENCV_QT_DIR="/usr/local/lib/python3.10/dist-packages/cv2/qt/plugins" \
    && if [ -d "$OPENCV_QT_DIR" ]; then rm -rf "$OPENCV_QT_DIR"; fi \
    && OPENCV_QT_DIR2="/usr/local/lib/python3.10/dist-packages/cv2/qt" \
    && if [ -d "$OPENCV_QT_DIR2" ]; then rm -rf "$OPENCV_QT_DIR2"; fi

# Install additional packages for visual servoing
RUN python3 -m pip install --no-cache-dir \
        open3d \
        --ignore-installed \
    && python3 -m pip install --no-cache-dir --upgrade \
        ros2_numpy

# Note: numpy/scipy/opencv-contrib are already pinned above; avoid uninstall/reinstall churn

# Final cleanup and version verification
RUN python3 -m pip cache purge

# Configure Qt environment to prevent plugin conflicts
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
ENV QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
ENV QT_DEBUG_PLUGINS=0
# Ensure OpenCV doesn't interfere with Qt
ENV OPENCV_QT_ENABLED=0


################################################################################################
# Final Release stage
# - Creates workspace and sets up environment
################################################################################################
FROM python-deps AS release

# Ensure HOME points to the non-root user's home in the final image
ENV HOME=/home/$USERNAME

# Optional: allow passing a GitHub token and repo info for stretch3 clone
ARG GITHUB_TOKEN=
ARG STRETCH3_REPO=https://github.com/hello-robot/stretch3.git
ARG STRETCH3_REF=main
ARG ENABLE_STRETCH3=0

# Create workspace structure
WORKDIR /workspace/ros2_ws
RUN mkdir -p src && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user for workspace setup
USER $USERNAME
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install || true

# Switch back to root for final setup
USER root
WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir --upgrade psutil

# Copy project files (if any)
COPY --chown=$USERNAME:$USERNAME . /workspace

# Build and install librealsense if present
RUN if [ -d /workspace/librealsense ]; then \
        cd /workspace/librealsense \
        && mkdir -p build && cd build \
        && cmake .. -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false \
        && make -j"$(nproc)" \
        && make install; \
    else \
        echo "/workspace/librealsense not found, skipping librealsense build"; \
    fi

# Setup environment variables in bashrc
RUN echo "" >> /root/.bashrc \
    && echo "# ROS2 and Stretch Robot Environment Setup" >> /root/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "export HELLO_FLEET_ID=stretch-se3-3092" >> /root/.bashrc \
    && echo "export HELLO_FLEET_PATH=/root/stretch_user" >> /root/.bashrc \
    && echo "if [ -f /workspace/ros2_ws/install/setup.bash ]; then" >> /root/.bashrc \
    && echo "    source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc \
    && echo "fi" >> /root/.bashrc \
    && echo "if [ -f /root/helloRobot_stretch3/colcon_ws/install/setup.bash ]; then" >> /root/.bashrc \
    && echo "    source /root/helloRobot_stretch3/colcon_ws/install/setup.bash" >> /root/.bashrc \
    && echo "fi" >> /root/.bashrc

# Ensure non-root user shells also source ROS 2 and overlays
RUN echo "" >> /home/$USERNAME/.bashrc \
    && echo "# ROS2 and Stretch Robot Environment Setup" >> /home/$USERNAME/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f /workspace/ros2_ws/install/setup.bash ]; then" >> /home/$USERNAME/.bashrc \
    && echo "    source /workspace/ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "fi" >> /home/$USERNAME/.bashrc \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

# Install additional system packages for visual servoing
RUN apt-get update && apt-get install -y \
        python3-pytest \
        whiptail \
    && rm -rf /var/lib/apt/lists/*

# Setup Stretch3 workspace and parameter files for visual servoing
# Clone stretch3 only when explicitly enabled; support optional token; do not fail build if unavailable
RUN if [ "${ENABLE_STRETCH3}" = "1" ]; then \
        if [ -n "$GITHUB_TOKEN" ]; then \
            git clone --depth 1 "https://${GITHUB_TOKEN}@github.com/hello-robot/stretch3.git" /root/stretch3 || true; \
        else \
            git clone --depth 1 "$STRETCH3_REPO" /root/stretch3 || true; \
        fi; \
    else \
        echo "Skipping stretch3 clone (ENABLE_STRETCH3=${ENABLE_STRETCH3})"; \
    fi \
    && mkdir -p /root/stretch_user/stretch-se3-3092 \
    && if [ -f /root/stretch3/stretch-se3-3092/stretch_user_params.yaml ]; then \
        ln -sf /root/stretch3/stretch-se3-3092/stretch_user_params.yaml /root/stretch_user/stretch-se3-3092/stretch_user_params.yaml; \
    fi \
    && if [ -f /root/stretch3/stretch-se3-3092/stretch_configuration_params.yaml ]; then \
        ln -sf /root/stretch3/stretch-se3-3092/stretch_configuration_params.yaml /root/stretch_user/stretch-se3-3092/stretch_configuration_params.yaml; \
    fi

# Add hello_helpers to Python path in bashrc for visual servoing
# Only export hello_helpers path if it exists
RUN if [ -d \
        /root/stretch3/ament_ws/install/hello_helpers/lib/python3.10/site-packages \
    ]; then \
        echo "export PYTHONPATH=/root/stretch3/ament_ws/install/hello_helpers/lib/python3.10/site-packages:\\$PYTHONPATH" >> /root/.bashrc; \
    fi

# Add Qt/GUI environment setup to bashrc
RUN echo "" >> /root/.bashrc \
    && echo "# Qt/GUI Environment Setup" >> /root/.bashrc \
    && echo "export QT_QPA_PLATFORM=xcb" >> /root/.bashrc \
    && echo "export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms" >> /root/.bashrc \
    && echo "export QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins" >> /root/.bashrc \
    && echo "export QT_DEBUG_PLUGINS=0" >> /root/.bashrc \
    && echo "export OPENCV_QT_ENABLED=0" >> /root/.bashrc \
    && echo "export QT_X11_NO_MITSHM=1" >> /root/.bashrc

# Create Qt fix script
RUN echo '#!/bin/bash' > /usr/local/bin/fix-qt.sh \
    && echo 'echo "Diagnosing Qt/GUI issues..."' >> /usr/local/bin/fix-qt.sh \
    && echo 'echo "DISPLAY: $DISPLAY"' >> /usr/local/bin/fix-qt.sh \
    && echo 'echo "QT_QPA_PLATFORM: $QT_QPA_PLATFORM"' >> /usr/local/bin/fix-qt.sh \
    && echo 'echo "Available Qt plugins:"' >> /usr/local/bin/fix-qt.sh \
    && echo 'ls -la /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/ 2>/dev/null || echo "Qt5 plugins not found"' >> /usr/local/bin/fix-qt.sh \
    && echo 'echo "Removing OpenCV Qt conflicts..."' >> /usr/local/bin/fix-qt.sh \
    && echo 'find /usr/local/lib/python*/dist-packages/cv2 -name "qt" -type d -exec rm -rf {} + 2>/dev/null || true' >> /usr/local/bin/fix-qt.sh \
    && echo 'echo "Testing X11 connection..."' >> /usr/local/bin/fix-qt.sh \
    && echo 'xeyes &' >> /usr/local/bin/fix-qt.sh \
    && echo 'XEYES_PID=$!' >> /usr/local/bin/fix-qt.sh \
    && echo 'sleep 2' >> /usr/local/bin/fix-qt.sh \
    && echo 'kill $XEYES_PID 2>/dev/null || true' >> /usr/local/bin/fix-qt.sh \
    && echo 'echo "Qt fix script completed"' >> /usr/local/bin/fix-qt.sh \
    && chmod +x /usr/local/bin/fix-qt.sh

# Switch back to non-root user as default
USER $USERNAME
WORKDIR /workspace

# Extra non-GPU utilities and Python packages (convenience/tools)
# - APT: common dev tools and CLI utilities
# - PIP: helpful runtime/dev libraries (CPU-only)
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        ninja-build \
        ccache \
        pkg-config \
        htop \
        tmux \
        tree \
        jq \
        rsync \
        openssh-client \
        netcat-openbsd \
        dnsutils \
        python3-venv \
        ripgrep \
    && sudo rm -rf /var/lib/apt/lists/*

# Keep pip installs lightweight and CPU-friendly
RUN python3 -m pip install --no-cache-dir --ignore-installed \
        requests \
        "pydantic>=2" \
        loguru \
        python-dotenv \
        onnxruntime \
        rich \
        typer \
        tqdm

# --- Add Qt/PyQt development packages and fix pandas optional deps ---
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    python3-pyqt5 \
    libxcb-render-util0 \
    libxcb-xinput0 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pandas optional dependencies (remove warnings)
RUN python3 -m pip install --no-cache-dir -U \
      "numexpr>=2.8.4" \
      "bottleneck>=1.3.6"

# --- Make stretch logs writable even if /root is RO at runtime ---
USER root
RUN mkdir -p /workspace/log/stretch_body_logger \
    && mkdir -p /root/stretch3 \
    && rm -rf /root/stretch3/log || true \
    && ln -s /workspace/log /root/stretch3/log \
    && chown -R $USERNAME:$USERNAME /workspace/log
ENV STRETCH_LOG_DIR=/workspace/log/stretch_body_logger

# --- PyQt5 + Ultralytics YOLO + ZeroMQ (for recv_and_yolo_d405_images.py) ---
USER $USERNAME
RUN python3 -m pip install --no-cache-dir \
      "PyQt5>=5.15.0" \
      "ultralytics==8.3.66" \
      "pyzmq>=25" \
      "shapely>=2.0" \
      "lapx>=0.5.9" \
      "scikit-image>=0.22"

# ---- Pin setuptools near the end (system-wide) ----
USER root
RUN python3 -m pip install --no-cache-dir --force-reinstall 'setuptools>=68,<80'
USER $USERNAME

# ---- Install wit package ----
RUN python3 -m pip install --no-cache-dir wit

# Default command
CMD ["bash"]
