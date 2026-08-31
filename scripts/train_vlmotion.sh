#!/usr/bin/env bash
set -euo pipefail

# VLMotion candidate-free semantic grounding:
#   full image + natural-language request -> variable-size coordinate set
#
# SAM3 is used only as a dense, trainable-through-adapters visual feature
# source.  No text detector, candidate cache, button prompt, panel ordering, or
# fixed output cardinality participates in training or inference.
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-wentao-yuan/robopoint-v1-vicuna-v1.5-13b}
export DATA_ROOT=${DATA_ROOT:-${WORKSPACE_ROOT}/training_vlmotion_100}
SPLIT_ROOT=${SPLIT_ROOT:-${DATA_ROOT}/split}
export OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE_ROOT}/checkpoints/vlmotion}
export EPOCHS=${EPOCHS:-5}
export LEARNING_RATE=${LEARNING_RATE:-5e-5}
export WARMUP_STEPS=${WARMUP_STEPS:-5}
export SAVE_STEPS=${SAVE_STEPS:-10}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
export MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-1024}

# These are deliberately false/empty. The auxiliary SAM3 heatmap is
# supervised directly by the union of annotated coordinates in train.py.
export SAM3_DYNAMIC_CANDIDATE_VIEW=false
export SAM3_CANDIDATE_CACHE=
export SAM3_INTENT_FOCUSED_TARGETS=false

if [[ ! -f "${SPLIT_ROOT}/train.json" ]]; then
    echo "VLMotion training split not found: ${SPLIT_ROOT}/train.json" >&2
    exit 1
fi
export DATA_FILE=${DATA_FILE:-${SPLIT_ROOT}/train.json}
export IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}/images}

exec "${WORKSPACE_ROOT}/scripts/train_vlmotion_docker.sh"
