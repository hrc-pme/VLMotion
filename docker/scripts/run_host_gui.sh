#!/bin/bash
# Host GUI + session node.
#
# ENABLE_BASE_MOTION=0  → service `vl`       (GUI on, bridge zeros cmd_vel)
# ENABLE_BASE_MOTION=1  → service `vlmotion` (GUI on, bridge forwards cmd_vel)
#
# Topics (Zenoh, same roles as StreamVLN, VLMotion names):
#   /camera/camera/color/image_raw/compressed          cams → GUI
#   /camera/camera/aligned_depth_to_color/image_raw    cams → host_session
#   /vlmotion/user_input            String   GUI → host
#   /vlmotion/run                   Bool     GUI → host
#   /vlmotion/target_pixel          Point    GUI → host_session
#   /vlmotion/enable_base_motion    Bool     host_session → bridge
#   /vlmotion/cmd_vel               Twist    host_session → bridge
#   /stretch/cmd_vel                Twist    bridge → stretch driver
#   /stretch3/odom                  Odometry driver → bridge

set -euo pipefail

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
WS_INSTALL="/workspace/ros2_ws/install/setup.bash"
if [ ! -f "${WS_INSTALL}" ]; then
  set -u
  echo "[run_host_gui] error: ${WS_INSTALL} not found." >&2
  echo "[run_host_gui] Build first: ./run.sh 4060ti cb" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${WS_INSTALL}"
set -u

export ENABLE_BASE_MOTION="${ENABLE_BASE_MOTION:-0}"
export VLMOTION_ROS_CAMERA=1
export MODEL_PATH="${MODEL_PATH:-wentao-yuan/robopoint-v1-vicuna-v1.5-13b}"

echo "[run_host_gui] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "[run_host_gui] RMW=${RMW_IMPLEMENTATION:-}"
echo "[run_host_gui] ENABLE_BASE_MOTION=${ENABLE_BASE_MOTION}"
echo "[run_host_gui] MODEL_PATH=${MODEL_PATH}"
echo "[run_host_gui] DISPLAY=${DISPLAY:-}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[run_host_gui] warning: DISPLAY is empty; GUI may fail to open" >&2
fi

exec ros2 launch vlmotion_host host.launch.py \
  "enable_base_motion:=${ENABLE_BASE_MOTION}" \
  "model_path:=${MODEL_PATH}" \
  "camera_image_topic:=${CAMERA_IMAGE_TOPIC:-/camera/camera/color/image_raw/compressed}" \
  "depth_image_topic:=${DEPTH_IMAGE_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
