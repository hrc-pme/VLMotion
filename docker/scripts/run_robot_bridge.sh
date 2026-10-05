#!/bin/bash
# Stretch3 bridge. Forwards /vlmotion/cmd_vel to /stretch/cmd_vel only while
# /vlmotion/enable_base_motion is true. The host sets that flag from
# service `vlmotion`.
#
# Camera is not republished here. hellorobot cams publish the D435i topics
# and the host GUI subscribes over Zenoh.
#
# Prerequisites on the robot, outside this container:
#   hellorobot driver + d435i, both on the same Zenoh router / ROS_DOMAIN_ID
# Host:
#   ./run.sh 4060ti zenoh-router
#   ./run.sh 4060ti vlmotion 30
#
# Publishes /stretch/cmd_vel only while enable_base_motion is true and the
# driver reports navigation mode (avoids stretch_driver cmd_vel errors).

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/zenoh/zenoh-env.sh"
ZENOH_HOST="${ZENOH_ROUTER_HOST:-$(zenoh_router_host)}"
ZENOH_PORT="${ZENOH_ROUTER_PORT:-$(zenoh_router_port)}"
if ! zenoh_wait_for_router "${ZENOH_HOST}" "${ZENOH_PORT}" 120; then
  echo "[run_robot_bridge] error: Zenoh router not reachable at ${ZENOH_HOST}:${ZENOH_PORT}" >&2
  echo "[run_robot_bridge] start host router: ./run.sh 4060ti zenoh-router" >&2
  exit 1
fi

echo "[run_robot_bridge] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "[run_robot_bridge] RMW=${RMW_IMPLEMENTATION:-}"
echo "[run_robot_bridge] odom=${ODOM_TOPIC:-/stretch3/odom}"
echo "[run_robot_bridge] cmd_vel=${CMD_VEL_TOPIC:-/stretch/cmd_vel}"

exec ros2 launch vlmotion_robot_bridge robot_bridge.launch.py \
  "odom_topic:=${ODOM_TOPIC:-/stretch3/odom}" \
  "cmd_vel_topic:=${CMD_VEL_TOPIC:-/stretch/cmd_vel}"
