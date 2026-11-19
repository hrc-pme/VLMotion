#!/usr/bin/env bash

set -eo pipefail

source /opt/ros/humble/setup.bash

echo "[vlmotion] RMW=${RMW_IMPLEMENTATION:-unset} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
echo "[vlmotion] Zenoh=${ZENOH_ROUTER_HOST:-unset}:${ZENOH_ROUTER_PORT:-unset} profile=${ZENOH_PROFILE:-unset}"
echo "[vlmotion] Controller=${VLMOTION_CONTROLLER_URL:-unset}"
echo "[vlmotion] Motion enabled=${VLMOTION_ENABLE_MOTION:-false}"
echo "[vlmotion] STT enabled=${VLMOTION_ENABLE_STT:-true} model=${VLMOTION_STT_MODEL_PATH:-small} cache=${VLMOTION_STT_MODEL_DIR:-/models/faster-whisper}"

router_host="${ZENOH_ROUTER_HOST:-127.0.0.1}"
router_port="${ZENOH_ROUTER_PORT:-7447}"
until nc -z "$router_host" "$router_port" >/dev/null 2>&1; do
  echo "[vlmotion] Waiting for Zenoh router tcp/${router_host}:${router_port} ..."
  sleep 2
done
echo "[vlmotion] Zenoh router is ready"

cd /workspace/ros2_ws
# The build/install directories are persistent Docker volumes. Clear only this
# generated package output so removed console scripts cannot survive upgrades.
rm -rf \
  /workspace/ros2_ws/build/white_point_pipeline \
  /workspace/ros2_ws/install/white_point_pipeline \
  /workspace/ros2_ws/build/vlmotion_stt \
  /workspace/ros2_ws/install/vlmotion_stt
colcon build \
  --symlink-install \
  --packages-select speech_recognition_msgs vlmotion_stt vlpoint white_point_pipeline \
  --cmake-args -DBUILD_TESTING=OFF
source /workspace/ros2_ws/install/setup.bash

exec ros2 launch white_point_pipeline hellorobot_app.launch.py \
  controller_url:="${VLMOTION_CONTROLLER_URL:-http://192.168.0.222:11000}" \
  model_path:="${VLMOTION_MODEL_PATH:-PME033541/vla13}" \
  enable_motion:="${VLMOTION_ENABLE_MOTION:-false}" \
  publish_odom_tf:="${VLMOTION_PUBLISH_ODOM_TF:-true}" \
  enable_stt:="${VLMOTION_ENABLE_STT:-true}" \
  stt_model_path:="${VLMOTION_STT_MODEL_PATH:-small}" \
  stt_model_dir:="${VLMOTION_STT_MODEL_DIR:-/models/faster-whisper}" \
  stt_language:="${VLMOTION_STT_LANGUAGE:-auto}" \
  stt_cpu_threads:="${VLMOTION_STT_CPU_THREADS:-8}"
