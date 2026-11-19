#!/usr/bin/env bash

# ROS 2/Zenoh client environment for the helloRobot_stretch3 application
# container. Values must match the hardware stack selected in run.sh.

domain_id="${1:-${ROS_DOMAIN_ID:-0}}"
if ! [[ "$domain_id" =~ ^[0-9]+$ ]] || ((10#$domain_id > 232)); then
  echo "ROS_DOMAIN_ID must be between 0 and 232" >&2
  return 1 2>/dev/null || exit 1
fi

export ROS_DOMAIN_ID="$domain_id"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ZENOH_PROFILE="${ZENOH_PROFILE:-local}"
export ZENOH_ROUTER_HOST="${ZENOH_ROUTER_HOST:-127.0.0.1}"
export ZENOH_ROUTER_PORT="${ZENOH_ROUTER_PORT:-7447}"
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-mode=\"client\";connect/endpoints=[\"tcp/${ZENOH_ROUTER_HOST}:${ZENOH_ROUTER_PORT}\"]}"

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
if [[ -f /workspace/ros2_ws/install/setup.bash ]]; then
  source /workspace/ros2_ws/install/setup.bash
fi

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "Zenoh router=tcp/${ZENOH_ROUTER_HOST}:${ZENOH_ROUTER_PORT} (${ZENOH_PROFILE})"
