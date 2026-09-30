#!/bin/bash
# Container entrypoint (stretch3): Zenoh client → host zenoh-router, then run command.
#
# Wired by docker/stretch3.compose.yaml for all services (dev, cb, bridge, ...).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/zenoh-env.sh"

ENDPOINT="$(zenoh_remote_client_endpoint)"
WAIT_HOST="$(zenoh_router_host)"
WAIT_PORT="$(zenoh_router_port)"

echo "[entrypoint-stretch3] role=client  connect=tcp/${ENDPOINT}"

if ! zenoh_wait_for_router "${WAIT_HOST}" "${WAIT_PORT}" 20; then
  echo "[entrypoint-stretch3] warning: router not reachable at ${ENDPOINT} yet (continuing)" >&2
  echo "[entrypoint-stretch3] ensure host is running: ./run.sh 4060ti zenoh-router" >&2
fi

zenoh_apply_client_env "${ENDPOINT}"
zenoh_persist_client_profile "entrypoint-stretch3.sh"

echo "[entrypoint-stretch3] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION}"
echo "[entrypoint-stretch3] ZENOH_CONFIG_OVERRIDE=${ZENOH_CONFIG_OVERRIDE}"

if [ "$#" -eq 0 ]; then
  set -- /bin/bash
fi
exec "$@"
