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
│   └── scripts/                   # Zenoh entrypoints, colcon build, GUI, bridge
└── ros2_ws/src/
    ├── vlpoint/                   # RoboPoint controller and worker
    ├── vlservo/                   # Qt GUI and visual servoing
    ├── vlmotion_host/             # pixel → /vlmotion/cmd_vel
    └── vlmotion_robot_bridge/     # gate onto /stretch/cmd_vel
```

## Requirements

- Ubuntu 22.04
- Docker, with the NVIDIA runtime on the 4060ti host
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
docker logs -f vlmotion-4060ti-vlmotion
docker logs -f vlmotion-stretch3-bridge
```

Stop a machine's containers with `./run.sh 4060ti stop` or `./run.sh stretch3 stop`.

The default model is [PME033541/vla13](https://huggingface.co/PME033541/vla13) at `/workspace/models/vla13` (`MODEL_PATH` in `.env`). The `vlmotion` service downloads it automatically if missing. To fetch manually:

```bash
hf download PME033541/vla13 --local-dir models/vla13 --exclude "runs/*"
```

## License

Apache-2.0
