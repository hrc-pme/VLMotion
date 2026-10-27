#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 fork_finetuning/check_cuda.py
python3 fork_finetuning/prepare_fork_coco.py \
  --dataset-root /workspace/Forks.v3i.coco \
  --category-id 1 \
  --prompt fork
python3 sam3/train/train.py \
  -c configs/fork/fork_bbox_smoke.yaml \
  --use-cluster 0 \
  --num-gpus 1
