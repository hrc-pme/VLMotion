# VLMotion application container

This repository is the application side of the Stretch system. Hardware is
owned by `/home/hrc/pomelo925/helloRobot_stretch3`; this container only runs:

- White Point GUI
- VLPoint HTTP client
- D435i/D405 pixel-to-3D conversion
- Offline faster-whisper speech-to-text
- Direct motion logic

The two containers exchange ROS 2 data through Zenoh. The GUI sends inference
requests over HTTP to `http://192.168.0.222:11000`.

## Start the hardware stack

Leave clearance around the arm before `cams-driver` homes the robot.

```bash
cd /home/hrc/pomelo925/helloRobot_stretch3
./run.sh local cams-driver 0
```

This starts the local Zenoh router, Stretch driver, D435i, and D405. Do not
start a second robot or RealSense driver from this repository.

Start the ReSpeaker hardware publisher as a second HelloRobot service. It can
run beside `cams-driver` and publishes `/speech_audio` to the same Zenoh router:

```bash
cd /home/hrc/pomelo925/helloRobot_stretch3
./run.sh local respeaker 0
```

HelloRobot owns the microphone device. The offline Whisper model and text
output are owned by VLMotion.

## Start VLMotion

In another terminal:

```bash
cd /home/hrc/VLMotion_stretch3_lbk
./docker/run.sh --build 0
```

The first build creates a small ROS application image. Later starts can omit
`--build`:

```bash
./docker/run.sh 0
```

Motion is disabled by default. The GUI, model connection, camera views, and
2D-to-3D nodes still run. Offline STT is enabled by default and uses the local
model cache mounted from `models/faster-whisper`.

The voice path is:

```text
ReSpeaker -> /speech_audio -> VLMotion Whisper
          -> /vlmotion/speech_to_text -> GUI
```

Use `./docker/run.sh --no-stt 0` only when speech recognition is not needed.

## Verify before enabling motion

Enter the application container:

```bash
./docker/run.sh --shell 0
```

Then verify the remote hardware data:

```bash
source /workspace/environment.sh 0
ros2 topic hz /head_camera/head_camera/color/image_raw
ros2 topic hz /head_camera/head_camera/aligned_depth_to_color/image_raw
ros2 topic hz /gripper_camera/gripper_camera/color/image_raw
ros2 topic echo /joint_states --once
ros2 topic echo /stretch3/odom --once
ros2 action info /stretch_controller/follow_joint_trajectory
ros2 run tf2_ros tf2_echo stretch3/base_link head_camera_color_optical_frame
ros2 run tf2_ros tf2_echo stretch3/base_link gripper_camera_color_optical_frame
```

The hardware stack currently publishes `/stretch3/odom` without broadcasting
its odometry TF. VLMotion therefore starts a small odometry-to-TF bridge. Set
`VLMOTION_PUBLISH_ODOM_TF=false` if another localization system owns that TF.

After the checks pass, restart with motion enabled:

```bash
./docker/stop.sh
./docker/run.sh --motion 0
```

Keep the emergency stop reachable and the arm workspace clear.

## Zenoh profiles

The default is the local router used by `./run.sh local cams-driver 0`:

```text
ROS_DOMAIN_ID=0
ZENOH_ROUTER_HOST=127.0.0.1
ZENOH_ROUTER_PORT=7447
```

To use the lab Spark router instead, both repositories must use the same
profile and domain:

```bash
ZENOH_PROFILE=spark \
ZENOH_ROUTER_HOST=192.168.0.230 \
./docker/run.sh 0
```

## Launch file

The application-only launch file is:

```bash
ros2 launch white_point_pipeline hellorobot_app.launch.py
```

It consumes these hardware topics by default:

```text
/head_camera/head_camera/color/image_raw/compressed
/head_camera/head_camera/aligned_depth_to_color/image_raw
/head_camera/head_camera/color/camera_info
/gripper_camera/gripper_camera/color/image_rect_raw/compressed
/gripper_camera/gripper_camera/aligned_depth_to_color/image_raw
/gripper_camera/gripper_camera/color/camera_info
/stretch3/tf
/stretch3/tf_static
/stretch3/odom
/joint_states
/speech_audio
```

It publishes commands only when `enable_motion:=true`:

```text
/stretch/cmd_vel
/stretch_controller/follow_joint_trajectory
```

The old standalone STT container is no longer required for VLMotion. The
bundled `vlmotion_stt` node uses the ignored `models/faster-whisper` cache.
