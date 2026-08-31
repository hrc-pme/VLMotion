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

## Training

The initial training setup uses 100 images sampled evenly from 20 different scene groups. Each image retains all associated annotations, such as up and down, resulting in 200 training and validation records in total.

### 1. Enter the Container

Run on the host:

```bash
docker exec -it vlmotion bash
```

Inside the container, verify the GPU and training data:

```bash
cd /workspace
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
find training_vlmotion_100/images -type f | wc -l
```

The output should include `True`, the GPU name, and an image count of `100`.

### 2. Start Training

```bash
cd /workspace
mkdir -p logs checkpoints
set -o pipefail
bash scripts/train_vlmotion.sh 2>&1 | tee logs/vlmotion-train.log
```

On the first run, the base model, CLIP, and SAM3 are downloaded to `/workspace/.cache/huggingface`. The corresponding location on the host is `/home/alan/VLMotion/.cache/huggingface`.

The training script directly uses the existing model code in `/workspace/ros2_ws/src/vlpoint`. `vlpoint` remains the existing ROS package name, while the public training scripts and outputs consistently use the `vlmotion` name.

Default training configuration:

- 100 images and 160 training records
- 20 images from 4 complete scene groups for validation, totaling 40 validation records
- 5 epochs
- Batch size 1 with 16 gradient accumulation steps
- Learning rate `5e-5`
- Approximately 50 optimizer steps, with a checkpoint saved every 10 steps
- BF16 and 4-bit QLoRA
- Output directory: `/workspace/checkpoints/vlmotion`

If the output directory already contains a `checkpoint-*` directory, running the same command again automatically resumes training.

### 3. Monitor or Stop Training

In another terminal, run:

```bash
docker exec -it vlmotion bash
tail -f /workspace/logs/vlmotion-train.log
```

You can also run `nvidia-smi` on the host. To stop training, press `Ctrl-C` in the training terminal. Any checkpoints that have already been written are preserved.

### 4. Merge the Model

After `/workspace/checkpoints/vlmotion/done.md` appears, run:

```bash
cd /workspace
bash scripts/merge_vlmotion.sh
```

The merged model is written to `/workspace/checkpoints/vlmotion-merged`. The merge loads the 13B model in FP16, so stop other GPU workloads first.

### 5. Change Training Parameters

Set parameters before the command. For example, to run a 10-step smoke test:

```bash
cd /workspace
OUTPUT_DIR=/workspace/checkpoints/vlmotion-smoke \
MAX_STEPS=10 \
SAVE_STEPS=10 \
bash scripts/train_vlmotion.sh
```

Do not reuse the smoke-test output directory for a full training run.

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
