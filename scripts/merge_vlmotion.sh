#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
BASE_MODEL=${BASE_MODEL:-wentao-yuan/robopoint-v1-vicuna-v1.5-13b}
ADAPTER_PATH=${ADAPTER_PATH:-${WORKSPACE_ROOT}/checkpoints/vlmotion}
MERGED_OUTPUT=${MERGED_OUTPUT:-${WORKSPACE_ROOT}/checkpoints/vlmotion-merged}
export PYTHONPATH=${WORKSPACE_ROOT}/ros2_ws/src/vlpoint:${PYTHONPATH:-}

python3 "${WORKSPACE_ROOT}/scripts/merge_vlmotion.py" \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER_PATH}" \
    --output "${MERGED_OUTPUT}" \
    --require-candidate-head
