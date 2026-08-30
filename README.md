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
└── environment.sh         # Environment setup script
```

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Docker (optional)
- Python 3.10+

## Installation and Setup

**Clone Repository**
```bash
git clone git@github.com:hrc-pme/VLMotion.git
cd VLMotion
 ```

SSH access to the hrc-pme/VLMotion repository must be configured on the local machine before cloning.

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
