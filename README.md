# VLMotion

Vision-language control for a Hello Robot Stretch 3. The GPU host runs the GUI and RoboPoint model. The robot runs a velocity bridge. The two machines share topics through Zenoh (`rmw_zenoh_cpp`), with the same host address as StreamVLN (`192.168.0.246:7447`).

`vlmotion` opens the GUI. The bridge forwards base velocity to `/stretch/cmd_vel` only after Start LLM Navigation, and only when a target point and depth are available.

## Layout

```
VLMotion/
├── run.sh                         # TUI / CLI (device, then service)
├── .env.template
├── docker/
│   ├── 4060ti.dockerfile          # CUDA 12.1 host image (RTX 4060 Ti)
│   ├── 4060ti.compose.yaml
│   ├── 5060.dockerfile            # CUDA 12.8 host image (RTX 5060)
│   ├── 5060.compose.yaml
│   ├── stretch3.dockerfile        # robot image
│   ├── stretch3.compose.yaml
│   ├── Dockerfile                 # upstream monolithic image (main)
│   ├── docker-compose.yml
│   └── scripts/                   # Zenoh entrypoints, colcon build, GUI, bridge
├── scripts/                       # training / merge (from upstream main)
├── training_vlmotion_100/         # sample training dataset
└── ros2_ws/src/
    ├── vlpoint/                   # RoboPoint controller and worker
    ├── vlservo/                   # Qt GUI and visual servoing
    ├── vlmotion_host/             # pixel → /vlmotion/cmd_vel
    ├── vlmotion_robot_bridge/     # gate onto /stretch/cmd_vel
    ├── audio_common/              # audio stack (upstream)
    └── audio_listener_worker/
```

## Requirements

- Ubuntu 22.04
- Docker, with the NVIDIA runtime on the GPU host
- `dialog` or `whiptail` for the TUI
- On the robot, the hellorobot driver and D435i already publishing on the same Zenoh domain

```bash
cp .env.template .env
```

## Services

Select a machine, then a service. `./run.sh` opens the TUI. The third argument is `ROS_DOMAIN_ID` (0–232, default 0).

| Machine | Services |
|---|---|
| `4060ti` | `zenoh-router`, `dev`, `cb`, `vlmotion`, `build`, `stop` |
| `5060` | same as `4060ti` (RTX 5060 / CUDA 12.8 image) |
| `stretch3` | `dev`, `cb`, `bridge`, `build`, `stop` |

`cb` runs `colcon build --symlink-install` inside the image for that machine. On GPU hosts it starts the Zenoh router first, because those containers are Zenoh clients.

```bash
./run.sh 4060ti build          # build the GPU image once
./run.sh 5060 build            # build the RTX 5060 image
./run.sh 4060ti cb             # colcon build on the host
./run.sh stretch3 build        # build the robot image once
./run.sh stretch3 cb           # colcon build on the robot
```

## Run

On a GPU host (`5060` or `4060ti`; `./run.sh` auto-detects RTX 5060), allow Docker to open windows, then start the router and a GUI:

```bash
xhost +local:docker
./run.sh 5060 zenoh-router
./run.sh 5060 vlmotion 30      # GUI; base moves after Start LLM Navigation
```

On the Stretch 3, after the driver and D435i are up:

```bash
./run.sh stretch3 bridge 30
```

Logs:

```bash
docker logs -f vlmotion-5060-vlmotion
docker logs -f vlmotion-stretch3-bridge
```

Stop a machine's containers with `./run.sh 5060 stop` or `./run.sh stretch3 stop`.

The default model is [PME033541/vla13](https://huggingface.co/PME033541/vla13) at `/workspace/models/vla13` (`MODEL_PATH` in `.env`). The `vlmotion` service downloads it automatically if missing. To fetch manually:

```bash
hf download PME033541/vla13 --local-dir models/vla13 --exclude "runs/*"
```

## Training

The initial training setup uses 100 images sampled evenly from 20 different scene groups. Each image retains all associated annotations, such as up and down, resulting in 200 training and validation records in total.

### 1. Enter the Container

Run on the host (monolithic `docker/` stack or a profile dev container):

```bash
docker exec -it vlmotion bash
```

Inside the container, verify the GPU and training data:

```bash
cd /workspace
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
find training_vlmotion_100/images -type f | wc -l
```

### 2. Start Training

```bash
cd /workspace
mkdir -p logs checkpoints
set -o pipefail
bash scripts/train_vlmotion.sh 2>&1 | tee logs/vlmotion-train.log
```

Default output directory: `/workspace/checkpoints/vlmotion`. See upstream docs in git history for full hyperparameters.

### 3. Merge the Model

After `/workspace/checkpoints/vlmotion/done.md` appears:

```bash
cd /workspace
bash scripts/merge_vlmotion.sh
```

## Development

```bash
cd ros2_ws
colcon build --symlink-install --packages-select vlpoint vlservo
source install/setup.bash
```

## License

Apache-2.0
