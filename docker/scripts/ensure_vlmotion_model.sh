#!/usr/bin/env bash
# Ensure MODEL_PATH contains a loadable Hugging Face checkpoint; download if not.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/vla13}"
VLMOTION_HF_MODEL_ID="${VLMOTION_HF_MODEL_ID:-PME033541/vla13}"

model_ready() {
  local dir=$1
  MODEL_CHECK_DIR="${dir}" python3 - <<'PY'
import json
import os
import sys

d = os.environ.get("MODEL_CHECK_DIR", "")
cfg = os.path.join(d, "config.json")
if not os.path.isdir(d) or not os.path.isfile(cfg):
    sys.exit(1)

single = ("model.safetensors", "pytorch_model.bin")
if any(os.path.isfile(os.path.join(d, name)) for name in single):
    sys.exit(0)

index = os.path.join(d, "model.safetensors.index.json")
if os.path.isfile(index):
    with open(index, encoding="utf-8") as f:
        shards = set(json.load(f).get("weight_map", {}).values())
    if shards and all(os.path.isfile(os.path.join(d, s)) for s in shards):
        sys.exit(0)

sys.exit(1)
PY
}

if model_ready "${MODEL_PATH}"; then
  echo "[ensure_vlmotion_model] checkpoint ok: ${MODEL_PATH}"
  exit 0
fi

echo "[ensure_vlmotion_model] missing or incomplete checkpoint at ${MODEL_PATH}"
echo "[ensure_vlmotion_model] downloading ${VLMOTION_HF_MODEL_ID} (exclude runs/*) ..."

mkdir -p "${MODEL_PATH}"

export VLMOTION_HF_MODEL_ID MODEL_PATH

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["VLMOTION_HF_MODEL_ID"]
local_dir = os.environ["MODEL_PATH"]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    ignore_patterns=["runs/*"],
)
print(f"[ensure_vlmotion_model] download complete: {local_dir}")
PY

if ! model_ready "${MODEL_PATH}"; then
  echo "[ensure_vlmotion_model] error: download finished but checkpoint still invalid at ${MODEL_PATH}" >&2
  exit 1
fi
