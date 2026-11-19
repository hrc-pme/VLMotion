#!/usr/bin/env bash

set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ZENOH_PROFILE="${ZENOH_PROFILE:-local}"
export ZENOH_ROUTER_HOST="${ZENOH_ROUTER_HOST:-127.0.0.1}"
export ZENOH_ROUTER_PORT="${ZENOH_ROUTER_PORT:-7447}"
export VLMOTION_ENABLE_MOTION="${VLMOTION_ENABLE_MOTION:-false}"
export VLMOTION_ENABLE_STT="${VLMOTION_ENABLE_STT:-true}"

build=false
follow=true
shell=false

usage() {
  cat <<'EOF'
Usage: ./docker/run.sh [--build] [--motion] [--no-stt] [--shell] [--no-follow] [ROS_DOMAIN_ID]

The default matches:
  /home/hrc/pomelo925/helloRobot_stretch3/run.sh local cams-driver 0

Options:
  --build      Force a Docker image rebuild.
  --motion     Enable the motion node. Verify camera and TF topics first.
  --no-stt     Disable the bundled offline speech-to-text node.
  --shell      Enter the running container after startup.
  --no-follow  Start in the background without following logs.
EOF
}

while (($#)); do
  case "$1" in
    --build) build=true ;;
    --motion) export VLMOTION_ENABLE_MOTION=true ;;
    --no-stt) export VLMOTION_ENABLE_STT=false ;;
    --shell) shell=true; follow=false ;;
    --no-follow) follow=false ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]] && ((10#$1 <= 232)); then
        export ROS_DOMAIN_ID="$1"
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
  shift
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed" >&2
  exit 1
fi

# Prepare X11 authorization before the GUI container starts.
xauth_file=/tmp/.docker.xauth
touch "$xauth_file"
xauth_list="$(xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' || true)"
if [[ -n "$xauth_list" ]]; then
  printf '%s\n' "$xauth_list" | xauth -f "$xauth_file" nmerge -
fi
chmod a+r "$xauth_file"
xhost +local:docker >/dev/null 2>&1 || true

compose=(docker compose -f docker/docker-compose.yml)
up_args=(up -d)
if [[ "$build" == true ]]; then
  up_args+=(--build)
fi

echo "Starting VLMotion: domain=${ROS_DOMAIN_ID} zenoh=${ZENOH_ROUTER_HOST}:${ZENOH_ROUTER_PORT} motion=${VLMOTION_ENABLE_MOTION} stt=${VLMOTION_ENABLE_STT}"
"${compose[@]}" "${up_args[@]}" vlmotion
"${compose[@]}" ps vlmotion

if [[ "$shell" == true ]]; then
  exec docker exec -it vlmotion bash
fi

if [[ "$follow" == true ]]; then
  exec "${compose[@]}" logs -f vlmotion
fi

echo "VLMotion is running. Logs: docker logs -f vlmotion"
