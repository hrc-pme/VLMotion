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

# setuptools>=80 dropped `develop --uninstall`, which colcon symlink-install
# still calls. setuptools>=68 needs packaging.canonicalize_version's
# strip_trailing_zero argument, which Ubuntu 22.04's packaging 21.3 lacks
# and packaging>=25 removed. Pin both here so an already-built image works
# without a rebuild.
python3 -m pip install --no-cache-dir --disable-pip-version-check \
  "setuptools>=68,<80" \
  "packaging>=24,<25"

cd "$WS_DIR"
colcon build --symlink-install "$@"
