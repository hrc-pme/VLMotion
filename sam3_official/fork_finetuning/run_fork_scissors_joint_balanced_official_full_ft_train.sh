#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="/workspace/dataset composition"
DATASET_ROOT="$REPO_ROOT/fork_finetuning/data/ForkScissors.joint.balanced.official.full_ft"

python3 "$REPO_ROOT/fork_finetuning/prepare_joint_fork_scissors_balanced.py" \
  --fork-source "$SOURCE_ROOT/Forks.coco" \
  --scissors-source "$SOURCE_ROOT/scissors.coco" \
  --output "$DATASET_ROOT" \
  --train-fraction 0.8 \
  --seed 42

for split in train test; do
  annotation="$DATASET_ROOT/$split/_annotations.coco.json"
  if [[ ! -s "$annotation" ]]; then
    echo "Missing or empty COCO annotation: $annotation" >&2
    exit 2
  fi
done

cd "$REPO_ROOT"
python3 fork_finetuning/check_cuda.py
python3 sam3/train/train.py \
  -c configs/fork/fork_scissors_joint_balanced_official_full_ft.yaml \
  --use-cluster 0 \
  --num-gpus 1
