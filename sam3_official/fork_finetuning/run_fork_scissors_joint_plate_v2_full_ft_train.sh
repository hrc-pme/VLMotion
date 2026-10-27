#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/.train_deps:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_ROOT="/workspace/dataset composition"
PLATE_ROOT="/workspace/Forks_plate.coco"
DATASET_ROOT="$REPO_ROOT/fork_finetuning/data/ForkScissors.joint.plate_v2.full_ft"
PREVIOUS_CHECKPOINT="$REPO_ROOT/fork_finetuning/experiments/fork_scissors_joint_balanced_official_full_ft/checkpoints/checkpoint.pt"

if [[ ! -s "$PREVIOUS_CHECKPOINT" ]]; then
  echo "Missing previous joint fine-tuned checkpoint: $PREVIOUS_CHECKPOINT" >&2
  exit 2
fi

python3 "$REPO_ROOT/fork_finetuning/prepare_joint_fork_scissors_balanced.py" \
  --fork-source "$SOURCE_ROOT/Forks.coco" \
  --scissors-source "$SOURCE_ROOT/scissors.coco" \
  --extra-fork-train-source "$PLATE_ROOT" \
  --output "$DATASET_ROOT" \
  --train-fraction 0.8 \
  --samples-per-concept 33 \
  --seed 42

python3 - "$PLATE_ROOT" "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

plate_root, dataset_root = map(Path, sys.argv[1:])
source = json.loads((plate_root / "train/_annotations.coco.json").read_text())
train = json.loads((dataset_root / "train/_annotations.coco.json").read_text())
test = json.loads((dataset_root / "test/_annotations.coco.json").read_text())
source_ids = {int(image["id"]) for image in source["images"]}
included = {
    int(image["source_image_id"])
    for image in train["images"]
    if image["file_name"].startswith("fork_plate_")
}
excluded = [
    image for image in test["images"]
    if image["file_name"].startswith("fork_plate_")
]
assert included == source_ids, f"Plate images missing from train: {source_ids - included}"
assert not excluded, "Plate images must not appear in test"
assert all((dataset_root / "train" / image["file_name"]).is_file()
           for image in train["images"]), "A train image symlink is broken"
assert len(train["images"]) == 55, "Unexpected train-image count"
assert len(test["images"]) == 13, "Unexpected test-image count"
print(f"Verified all {len(included)} plate images are in train; none are in test")
PY

cd "$REPO_ROOT"
python3 fork_finetuning/check_cuda.py
python3 sam3/train/train.py \
  -c configs/fork/fork_scissors_joint_plate_v2_full_ft.yaml \
  --use-cluster 0 \
  --num-gpus 1
