#!/bin/bash
# Host GUI + session node for service `vlmotion`.
# host_session calls /switch_to_navigation_mode once the Stretch driver is reachable.
#
# ENABLE_BASE_MOTION=1 lets the bridge forward /vlmotion/cmd_vel. The host
# still publishes zeros until Start LLM Navigation has a target and depth.
#
# Camera aliases (GUI selector; hellorobot keeps these topic names):
#   top camera     = D415  /camera_top/camera_top/...
#   head camera    = D435i /head_camera/head_camera/...
#   gripper camera = D405  color: .../color/image_rect_raw/compressed
#   LLM Grasping requires selecting gripper camera first; switching drives wrist
#   pose so the D405 view matches arm extend/retract (see stretch3 /tf).
#   /vlmotion/camera_select         String   GUI → host_session
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
export MODEL_PATH="${MODEL_PATH:-/workspace/models/vla13}"
export VLMOTION_HF_MODEL_ID="${VLMOTION_HF_MODEL_ID:-PME033541/vla13}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/ensure_vlmotion_model.sh"

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
  "model_path:=${MODEL_PATH}"
