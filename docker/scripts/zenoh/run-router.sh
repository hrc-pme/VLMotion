#!/bin/bash
# Dedicated Zenoh router process for compose service `zenoh-router`.
# One router per host; all other containers connect as clients.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/zenoh-env.sh"

zenoh_source_ros
zenoh_apply_router_env

PORT="$(zenoh_router_port)"
HOST="$(zenoh_router_host)"

echo "[zenoh-router] role=router"
echo "[zenoh-router] listen=tcp/0.0.0.0:${PORT}  (LAN advertise host=${HOST})"
echo "[zenoh-router] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION}"
echo "[zenoh-router] starting: ros2 run rmw_zenoh_cpp rmw_zenohd"

exec ros2 run rmw_zenoh_cpp rmw_zenohd
