#!/bin/bash
set -euo pipefail

# ROS setup.bash references optional unset vars under `set -u`.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

WS_DIR="/workspace/ros2_ws"

if [ ! -d "$WS_DIR/src" ]; then
  echo "Error: $WS_DIR/src not found." >&2
  echo "Create packages under ros2_ws/src/ before running colcon build." >&2
  exit 1
fi

cd "$WS_DIR"
colcon build --symlink-install "$@"
