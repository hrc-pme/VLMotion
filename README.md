# VLMotion

VLMotion is a ROS2-based vision-language robot control system that integrates two main packages: VLPoint and VLServo.

## Project Structure

```
VLMotion/
├── docker/                 # Docker container configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── build.sh           # Build Docker image
│   ├── run.sh             # Start Docker container
│   └── stop.sh            # Stop Docker container
├── ros2_ws/               # ROS2 workspace
│   └── src/
│       ├── vlpoint/       # VLPoint package (controller and worker)
│       └── vlservo/       # VLServo package (visual servoing)
├── sam3_official/         # SAM3 training repository
│   └── fork_finetuning/   # Fork/scissors fine-tuning scripts and outputs
└── environment.sh         # Environment setup script
```

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Docker (optional)
- Python 3.10+

## Installation and Setup

### Method 1: Using Docker (Recommended)

1. **Build Docker Image**
   ```bash
   cd docker
   ./build.sh
   ```

2. **Start Docker Container**
   ```bash
   ./run.sh
   ```

3. **Inside the Container, Setup Environment**
   ```bash
   source /workspace/environment.sh [ROS_DOMAIN_ID]
   ```
   - `ROS_DOMAIN_ID` is optional, defaults to 0, valid range: 0-232

4. **Build ROS2 Packages**
   ```bash
   cd /workspace/ros2_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

### Method 2: Local Installation

1. **Install ROS2 Humble**
   ```bash
   # See official documentation: https://docs.ros.org/en/humble/Installation.html
   ```

2. **Setup Environment**
   ```bash
   source environment.sh [ROS_DOMAIN_ID]
   ```

3. **Build ROS2 Packages**
   ```bash
   cd ros2_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

## Usage

### Launch VLPoint Package

The VLPoint package contains two main components: controller and worker.

#### 1. Launch Controller
```bash
ros2 launch vlpoint controller.launch.py
```

#### 2. Launch Worker
```bash
ros2 launch vlpoint worker.launch.py
```

#### 3. Launch Complete System (Controller + Worker)
```bash
ros2 launch vlpoint vlpoint.launch.py
```

### Launch VLServo Package

The VLServo package provides visual servoing functionality.

```bash
ros2 launch vlservo vlservoing.launch.py
```

## SAM3 Fine-tuning

Fork/scissors SAM3 微調的完整說明請參考：

- [Fine-tuning README](sam3_official/fork_finetuning/README.md)

目前使用的主要腳本與設定：

| 用途 | 位置 |
| --- | --- |
| 一鍵資料準備與完整訓練 | [`sam3_official/fork_finetuning/run_fork_scissors_joint_balanced_official_full_ft_train.sh`](sam3_official/fork_finetuning/run_fork_scissors_joint_balanced_official_full_ft_train.sh) |
| Fork/scissors dataset 準備 | [`sam3_official/fork_finetuning/prepare_joint_fork_scissors_balanced.py`](sam3_official/fork_finetuning/prepare_joint_fork_scissors_balanced.py) |
| 訓練設定 | [`sam3_official/sam3/train/configs/fork/fork_scissors_joint_balanced_official_full_ft.yaml`](sam3_official/sam3/train/configs/fork/fork_scissors_joint_balanced_official_full_ft.yaml) |
| 將 native checkpoint 匯出成 worker 格式 | [`sam3_official/fork_finetuning/export_finetuned_hf.py`](sam3_official/fork_finetuning/export_finetuned_hf.py) |

完整訓練入口：

```bash
bash /workspace/sam3_official/fork_finetuning/run_fork_scissors_joint_balanced_official_full_ft_train.sh
```

Native checkpoint、Hugging Face 權重儲存位置，以及如何替換到 worker，皆記錄於上述 fine-tuning README。

## Typical Workflow

### Full System Launch

Execute the following commands in **separate terminal windows**:

**Terminal 1 - Launch VLPoint Controller:**
```bash
source environment.sh
cd ros2_ws
source install/setup.bash
ros2 launch vlpoint controller.launch.py
```

**Terminal 2 - Launch VLPoint Worker:**
```bash
source environment.sh
cd ros2_ws
source install/setup.bash
ros2 launch vlpoint worker.launch.py
```

**Terminal 3 - Launch VLServo Visual Servoing:**
```bash
source environment.sh
cd ros2_ws
source install/setup.bash
ros2 launch vlservo vlservoing.launch.py
```

### Launch in Docker Environment

If using Docker, you can open multiple terminals inside the container:

```bash
# On host machine
docker exec -it vlmotion_container bash

# Inside container
source /workspace/environment.sh
cd /workspace/ros2_ws
source install/setup.bash
# Then execute the corresponding launch commands
```

## Package Description

### VLPoint
- **controller.launch.py**: Launches the main controller node for system coordination
- **worker.launch.py**: Launches the worker node for vision-language tasks
- **vlpoint.launch.py**: Launches both controller and worker simultaneously

### VLServo
- **vlservoing.launch.py**: Launches the visual servoing system for robot vision-based navigation and control

## Environment Variables

- `ROS_DOMAIN_ID`: ROS2 domain ID (0-232), used for multi-robot or multi-system isolation
- `ROS_DISTRO`: ROS2 distribution version (default: humble)
- `PYTHONWARNINGS`: Python warning filter settings
- `PIP_DISABLE_PIP_VERSION_CHECK`: Disable pip version check

## Troubleshooting

### Common Issues

1. **Package Not Found**
   ```bash
   # Make sure packages are built and environment is sourced
   cd ros2_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

2. **Permission Issues**
   ```bash
   # Check file permissions
   sudo chmod +x docker/*.sh
   ```

3. **ROS2 Communication Issues**
   ```bash
   # Check if ROS_DOMAIN_ID is consistent
   echo $ROS_DOMAIN_ID
   
   # Reset environment
   source environment.sh [DOMAIN_ID]
   ```

4. **GUI Issues in Docker Container**
   ```bash
   # Make sure X11 forwarding is enabled
   xhost +local:docker
   ```

## Development

### Rebuild Packages

```bash
cd ros2_ws
colcon build --symlink-install --packages-select vlpoint vlservo
source install/setup.bash
```

### Clean Build Artifacts

```bash
cd ros2_ws
rm -rf build/ install/ log/
```

## License

Apache-2.0
