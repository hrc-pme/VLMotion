#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/.train_deps:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="$REPO_ROOT/fork_finetuning/data/ForkScissors.joint.plate_v2.full_ft"
BASE_CHECKPOINT="$REPO_ROOT/checkpoints/sam3.pt"

if [[ ! -s "$BASE_CHECKPOINT" ]]; then
  echo "Missing original SAM3 checkpoint: $BASE_CHECKPOINT" >&2
  exit 2
fi

python3 - "$DATASET_ROOT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
train = json.loads((root / "train/_annotations.coco.json").read_text())
test = json.loads((root / "test/_annotations.coco.json").read_text())
plate = [image for image in train["images"] if image["file_name"].startswith("fork_plate_")]
assert len(plate) == 5, "All five plate images must be in train"
assert not any(image["file_name"].startswith("fork_plate_") for image in test["images"])
assert Counter(image["concept"] for image in train["images"]) == {"fork": 33, "scissors": 22}
plate_ids = {image["id"] for image in plate}
plate_annotations = [ann for ann in train["annotations"] if ann["image_id"] in plate_ids]
assert len(plate_annotations) == 45
assert all(ann["category_id"] == 1 and ann["noun_phrase"] == "fork" for ann in plate_annotations)
assert all((root / "train" / image["file_name"]).is_file() for image in train["images"])
print("Verified joint dataset: 33 fork + 22 scissors train images; all 5 plate images labeled fork")
PY

cd "$REPO_ROOT"
python3 fork_finetuning/check_cuda.py
python3 sam3/train/train.py \
  -c configs/fork/fork_scissors_joint_plate_v4_20ep_full_ft.yaml \
  --use-cluster 0 \
  --num-gpus 1
