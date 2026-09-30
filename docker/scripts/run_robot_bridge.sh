#!/bin/bash
# Stretch3 bridge. Forwards /vlmotion/cmd_vel to /stretch/cmd_vel only while
# /vlmotion/enable_base_motion is true (service vlmotion). Service vl keeps
# the base still.
#
# Camera is not republished here. hellorobot cams publish the D435i topics
# and the host GUI subscribes over Zenoh.
#
# Prerequisites on the robot, outside this container:
#   hellorobot driver + d435i, both on the same Zenoh router / ROS_DOMAIN_ID
# Host:
#   ./run.sh 4060ti zenoh-router
#   ./run.sh 4060ti vl 30
#   ./run.sh 4060ti vlmotion 30

set -euo pipefail

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
WS_INSTALL="/workspace/ros2_ws/install/setup.bash"
if [ ! -f "${WS_INSTALL}" ]; then
  set -u
  echo "[run_robot_bridge] error: ${WS_INSTALL} not found." >&2
  echo "[run_robot_bridge] Build first: ./run.sh stretch3 cb" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${WS_INSTALL}"
set -u

echo "[run_robot_bridge] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "[run_robot_bridge] RMW=${RMW_IMPLEMENTATION:-}"
echo "[run_robot_bridge] odom=${ODOM_TOPIC:-/stretch3/odom}"
echo "[run_robot_bridge] cmd_vel=${CMD_VEL_TOPIC:-/stretch/cmd_vel}"

exec ros2 launch vlmotion_robot_bridge robot_bridge.launch.py \
  "odom_topic:=${ODOM_TOPIC:-/stretch3/odom}" \
  "cmd_vel_topic:=${CMD_VEL_TOPIC:-/stretch/cmd_vel}"
