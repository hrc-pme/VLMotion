# Minimal LLM + ROS 2 Humble (GPU-ready)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Base system packages (add common build and multimedia deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-zmq \
    python3-opencv \
    git \
    curl \
    ca-certificates \
    gnupg2 \
    lsb-release \
    locales \
    sudo \
    build-essential \
    software-properties-common \
    libasound2-dev \
    portaudio19-dev \
    python3-pyaudio \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    unzip \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    python3-pyqt5 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xfixes0 \
    x11-apps \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 apt repository and key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Humble and ROS tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-desktop-full \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-argcomplete \
      whiptail \
      iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Locale and ROS env
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=humble \
    YOLO_CONFIG_DIR=/workspace/.ultralytics \
    ROS_DOMAIN_ID=0 \
    ROS_LOCALHOST_ONLY=0

# Initialize rosdep database
RUN rosdep init 2>/dev/null || true && rosdep update

# Use bash for subsequent RUN commands so we can 'source'
SHELL ["/bin/bash", "-lc"]

# Auto-source ROS environment for interactive shells
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh \${ROS_DOMAIN_ID:-0}; fi" >> /root/.bashrc && \
    echo "export PS1='\\u@\\h:\\w# '" >> /root/.bashrc

# Ensure login shells also load .bashrc (for docker exec -it bash -l)
RUN printf '%s\n' "if [ -f ~/.bashrc ]; then" \
    ". ~/.bashrc" \
    "fi" > /root/.bash_profile

# Python deps
# - 先行安裝 triton 依賴：sympy / mpmath，避免卸載 apt 版 sympy 失敗
# - Torch 使用 CUDA 12.1 的輪檔
# - Transformers 與影像處理相關套件
RUN pip3 install 'pip<25' 'setuptools<80,>=68' 'wheel>=0.40' 'packaging>=24.0' && \
    # 以 constraints 鎖定 numpy<2，避免後續安裝升級為 2.x
    printf "numpy==1.26.4\nsympy==1.14.0\nmpmath>=1.3.0\n" > /tmp/constraints.txt && \
    # 先安裝 numpy 1.26.4；對於 distutils 的 sympy，使用 --ignore-installed 避免卸載失敗
    pip3 install -c /tmp/constraints.txt numpy==1.26.4 && \
    pip3 install --no-cache-dir 'scipy>=1.11,<1.12' && \
    pip3 install --ignore-installed -c /tmp/constraints.txt sympy==1.14.0 "mpmath>=1.3.0" && \
    pip3 install -c /tmp/constraints.txt --index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision torchaudio && \
    pip3 install -c /tmp/constraints.txt \
      transformers \
      accelerate \
      bitsandbytes \
      soundfile \
      ultralytics \
      urchin \
      iopath \
      timm \
      ftfy==6.1.1 \
      einops \
      pycocotools

WORKDIR /workspace

# Copy entire project into the image
COPY . /workspace

# Bootstrap a ros2_ws (non-fatal if empty)
RUN mkdir -p /workspace/ros2_ws/src && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /workspace/ros2_ws && colcon build --symlink-install || true

# Additional Python packages (combine to one layer)
RUN pip3 install -c /tmp/constraints.txt \
    sentencepiece \
    protobuf \
    "Pillow>=10.0.0,<11.0.0" \
    wit \
    websocket-client \
    fastapi==0.112.4 \
    uvicorn[standard] \
    gradio==4.16.0 \
    scikit-image \
    scikit-learn \
    PyQt5>=5.15.0

# Keep container alive for interactive work (no extra shell)
CMD ["sleep", "infinity"]
