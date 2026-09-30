# VLMotion

Vision-language control for a Hello Robot Stretch 3. The GPU host runs the GUI and RoboPoint model. The robot runs a velocity bridge. The two machines share topics through Zenoh (`rmw_zenoh_cpp`), with the same host address as StreamVLN (`192.168.0.246:7447`).

`vlmotion` opens the GUI. The bridge forwards base velocity to `/stretch/cmd_vel` only after Start LLM Navigation, and only when a target point and depth are available.

## Layout

```
VLMotion/
├── run.sh                         # TUI / CLI (device, then service)
├── .env.template
├── docker/
│   ├── 4060ti.dockerfile          # CUDA 12.1 host image
│   ├── 4060ti.compose.yaml
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
| `stretch3` | `dev`, `cb`, `bridge`, `build`, `stop` |

`cb` runs `colcon build --symlink-install` inside the image for that machine. On the 4060ti it starts the Zenoh router first, because that container is a Zenoh client.

```bash
./run.sh 4060ti build          # build the GPU image once
./run.sh 4060ti cb             # colcon build on the host
./run.sh stretch3 build        # build the robot image once
./run.sh stretch3 cb           # colcon build on the robot
```

## Run

On the 4060ti, allow Docker to open windows, then start the router and a GUI:

```bash
xhost +local:docker
./run.sh 4060ti zenoh-router
./run.sh 4060ti vlmotion 30    # GUI; base moves after Start LLM Navigation
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

## Topics

Camera and odometry come from the hellorobot driver. The bridge does not republish images.

| Topic | Type | Path |
|---|---|---|
| `/camera_top/camera_top/color/image_raw/compressed` | CompressedImage | D415 top camera → GUI |
| `/camera_top/camera_top/aligned_depth_to_color/image_raw` | Image | D415 depth, only if that stream is enabled |
| `/head_camera/head_camera/color/image_raw/compressed` | CompressedImage | D435i head camera → GUI |
| `/head_camera/head_camera/aligned_depth_to_color/image_raw` | Image | D435i depth → host |
| `/vlmotion/camera_select` | String | GUI → host (`top camera` or `head camera`) |
| `/vlmotion/user_input` | String | GUI → host |
| `/vlmotion/run` | Bool | GUI → host |
| `/vlmotion/target_pixel` | Point | GUI → host |
| `/vlmotion/enable_base_motion` | Bool | host → bridge |
| `/vlmotion/cmd_vel` | Twist | host → bridge |
| `/stretch/cmd_vel` | Twist | bridge → driver (zeros until navigation is running) |
| `/stretch3/odom` | Odometry | driver → bridge |

The default model is `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` (`MODEL_PATH` in `.env`).

## License

Apache-2.0
