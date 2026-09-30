#!/bin/bash
# 4060ti host entrypoint: Zenoh **client** only (router is compose service zenoh-router).
#
# Wired by docker/4060ti.compose.yaml for dev / cb / infer.
# Connects to ZENOH_HOST_CLIENT_ENDPOINT (default 127.0.0.1:7447) on host network.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/zenoh-env.sh"

ENDPOINT="$(zenoh_host_client_endpoint)"
# Endpoint is host:port — split for wait check.
WAIT_HOST="${ENDPOINT%%:*}"
WAIT_PORT="${ENDPOINT##*:}"

echo "[entrypoint-4060ti] role=client  connect=tcp/${ENDPOINT}"

if ! zenoh_wait_for_router "${WAIT_HOST}" "${WAIT_PORT}" 40; then
  echo "[entrypoint-4060ti] error: Zenoh router not reachable at ${ENDPOINT}" >&2
  echo "[entrypoint-4060ti] start it first: ./run.sh 4060ti zenoh-router" >&2
  exit 1
fi

zenoh_apply_client_env "${ENDPOINT}"
zenoh_persist_client_profile "entrypoint-4060ti.sh"

echo "[entrypoint-4060ti] router ok  ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION}"
echo "[entrypoint-4060ti] ZENOH_CONFIG_OVERRIDE=${ZENOH_CONFIG_OVERRIDE}"

if [ "$#" -eq 0 ]; then
  set -- /bin/bash
fi
exec "$@"
