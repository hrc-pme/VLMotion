# VLMotion

VLMotion is a ROS 2 project for the Hello Robot Stretch. It includes the White Point motion pipeline, navigation and Unity tools, mapping utilities, and a rosbag recorder for dataset preparation.

The commands below assume Docker is used and the repository is mounted at `/workspace`.

## Docker Setup

The Docker configuration uses host networking, exposes the robot and camera devices, mounts the ROS 2 workspace and output folders, and supports X11 applications.

Before building, check the Stretch fleet ID and calibration path in `docker/docker-compose.yml`. They currently use `stretch-se3-3092` and `${HOME}/stretch3/stretch-se3-3092`.

Build the image and start the container from the repository root:

```bash
./docker/build.sh
```

To rebuild without cache:

```bash
./docker/build.sh --no-cache
```

Enter the container with ROS domain ID `0`:

```bash
./docker/run.sh 0
```

Build and source the ROS 2 workspace inside the container:

```bash
source /workspace/environment.sh 0
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
cd /workspace
```

Run the following in each new terminal:

```bash
source /workspace/environment.sh 0
source /workspace/ros2_ws/install/setup.bash
```

Stop the container with:

```bash
./docker/stop.sh
```

## White Point Pipeline

This launches the Stretch driver, RealSense cameras, camera transforms, White Point GUI, 3D point conversion, and direct motion controller. D405 and ReSpeaker support are enabled by default.

```bash
ros2 launch white_point_pipeline white_point_direct_pipeline.launch.py
```

For a D435i-only setup without ReSpeaker:

```bash
ros2 launch white_point_pipeline white_point_direct_pipeline.launch.py \
  enable_d405_view:=false \
  enable_respeaker:=false
```

Useful launch arguments include `controller_url`, `model_path`, `use_compressed_color`, and `require_point_confirmation`.

## Camera Setup

The dataset recorder does not start the camera. The White Point launch file starts `realsense2_camera_node` with the node and camera name `d435i`. Its serial number and topic prefix are defined in `ros2_ws/src/white_point_pipeline/config/cameras.yaml`.

The pipeline enables RGB, depth, aligned depth, TF, and point-cloud output at `640x480x30`. Do not start another D435i node while the pipeline is running.

To start only the robot driver and D435i for dataset collection, use two terminals:

```bash
# Terminal 1: robot state and TF
ros2 launch stretch_launch stretch_driver.launch.py \
  mode:=navigation broadcast_odom_tf:=True
```

```bash
# Terminal 2: D435i camera
ros2 launch stretch_launch d435i_basic.launch.py \
  camera_namespace:=/ camera_name:=d435i
```

Check the required topics before recording:

```bash
ros2 topic hz /d435i/color/image_raw/compressed
ros2 topic hz /d435i/depth/image_rect_raw/compressedDepth
```

## Dataset Recording

`record_with_ui.launch.py` opens a small Tkinter interface that runs `ros2 bag record`. It records raw ROS data for later conversion or annotation; it does not start the camera or create dataset labels.

The default configuration records:

- `/tf_static`
- `/d435i/color/image_raw/compressed`
- `/d435i/depth/image_rect_raw/compressedDepth`

Start the White Point pipeline, or start the driver and camera separately, then launch the recorder in another terminal:

```bash
ros2 launch bag_recorder record_with_ui.launch.py
```

Use **Start Recording** and **Stop** in the UI. Bags are saved as numbered folders such as `0001` and `0002` under:

```text
/workspace/Outputs/rosbags/test
```

To use a custom configuration:

```bash
ros2 launch bag_recorder record_with_ui.launch.py \
  config_file:=/workspace/my_bag_recorder.yaml
```

The recorder can also run without a GUI:

```bash
ros2 launch bag_recorder record.launch.py
```

Inspect or replay a recording with:

```bash
ros2 bag info /workspace/Outputs/rosbags/test/0001
ros2 bag play /workspace/Outputs/rosbags/test/0001
```

## Navigation

Start the RPLIDAR driver and scan filter:

```bash
ros2 launch stretch_launch rplidar.launch.py
```

Start Nav2 with one of the available maps:

```bash
ros2 launch stretch_launch navigation.launch.py map:=maps/elevator_6f.yaml
```

```bash
ros2 launch stretch_launch navigation.launch.py map:=maps/6f_ele2.yaml
```

Publish the `map -> base_link` transform as a `PoseStamped` message on `/pose` for Unity or other clients:

```bash
ros2 run stretch_unity_bridge tf_to_pose
```

## Unity

Start rosbridge WebSocket using the default port or port `8080`:

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=8080
```

Start the Unity ROS TCP endpoint using its defaults or port `10000`:

```bash
ros2 launch ros_tcp_endpoint endpoint.py
ros2 launch ros_tcp_endpoint endpoint.py tcp_ip:=0.0.0.0 tcp_port:=10000
```

Run the bridge that converts Unity joint commands into Stretch commands:

```bash
ros2 run stretch_unity_bridge stretch_unity_bridge_joint_states
```

Only run one command from each pair of alternative rosbridge and TCP endpoint configurations.

## Mapping

Start the Stretch driver, RPLIDAR, teleoperation, SLAM Toolbox, and RViz:

```bash
ros2 launch stretch_launch offline_mapping.launch.py
```

Save the completed map:

```bash
cd /workspace/maps
ros2 run nav2_map_server map_saver_cli -f ./<map_name>
```

## Stretch Services

These services require the Stretch driver or White Point pipeline to be running.

```bash
# Return control to the gamepad
ros2 service call /switch_to_gamepad_mode std_srvs/srv/Trigger {}

# Enable navigation control
ros2 service call /switch_to_navigation_mode std_srvs/srv/Trigger {}

# Home the robot joints
ros2 service call /home_the_robot std_srvs/srv/Trigger {}
```

Make sure the robot has enough free space before running the homing service.
