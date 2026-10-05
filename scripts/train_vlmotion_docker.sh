#!/usr/bin/env bash
set -euo pipefail

# Single-GPU SAM3 + CLIP spatial-relation fine-tuning for the `alan` container.
# Every path and major hyperparameter can be overridden from the environment.

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
MODEL_CODE_ROOT=${MODEL_CODE_ROOT:-${WORKSPACE_ROOT}/ros2_ws/src/vlpoint}
DATA_ROOT=${DATA_ROOT:-${WORKSPACE_ROOT}/training_vlmotion_100}
DATA_FILE=${DATA_FILE:-${DATA_ROOT}/labeled_dataset.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}/images}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE_ROOT}/checkpoints/vlmotion}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-wentao-yuan/robopoint-v1-vicuna-v1.5-13b}
CLIP_VISION_TOWER=${CLIP_VISION_TOWER:-openai/clip-vit-large-patch14-336}
SAM3_VISION_TOWER=${SAM3_VISION_TOWER:-facebook/sam3}
CUDA_DEVICE=${CUDA_DEVICE:-0}

EPOCHS=${EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
SAVE_STEPS=${SAVE_STEPS:-250}
LOGGING_STEPS=${LOGGING_STEPS:-5}
WARMUP_STEPS=${WARMUP_STEPS:-100}
NUM_WORKERS=${NUM_WORKERS:-2}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-1024}
REPORT_TO=${REPORT_TO:-none}
TUNE_MM_MLP_ADAPTER=${TUNE_MM_MLP_ADAPTER:-false}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-1e-5}
SAM3_DYNAMIC_CANDIDATE_VIEW=${SAM3_DYNAMIC_CANDIDATE_VIEW:-false}
SAM3_CANDIDATE_CACHE=${SAM3_CANDIDATE_CACHE:-}
SAM3_MAX_CANDIDATES=${SAM3_MAX_CANDIDATES:-16}
SAM3_INTENT_FOCUSED_TARGETS=${SAM3_INTENT_FOCUSED_TARGETS:-false}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
export HF_HOME=${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export PYTHONPATH=${MODEL_CODE_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Training JSON not found: ${DATA_FILE}" >&2
    exit 1
fi
if [[ ! -d "${IMAGE_FOLDER}" ]]; then
    echo "Training image folder not found: ${IMAGE_FOLDER}" >&2
    exit 1
fi

DYNAMIC_CANDIDATE_ARGS=()
if [[ "${SAM3_DYNAMIC_CANDIDATE_VIEW}" == "true" ]]; then
    if [[ -z "${SAM3_CANDIDATE_CACHE}" || ! -f "${SAM3_CANDIDATE_CACHE}" ]]; then
        echo "SAM3 candidate cache not found: ${SAM3_CANDIDATE_CACHE}" >&2
        exit 1
    fi
    DYNAMIC_CANDIDATE_ARGS=(
        --sam3_dynamic_candidate_view true
        --sam3_candidate_cache "${SAM3_CANDIDATE_CACHE}"
        --sam3_max_candidates "${SAM3_MAX_CANDIDATES}"
        --sam3_intent_focused_targets "${SAM3_INTENT_FOCUSED_TARGETS}"
    )
fi

python3 - <<'PY'
import importlib.util
import sys
import torch

missing = [name for name in ("peft", "bitsandbytes") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing training packages: " + ", ".join(missing), file=sys.stderr)
    print("Install them inside the container with: pip3 install peft bitsandbytes", file=sys.stderr)
    raise SystemExit(2)
if not torch.cuda.is_available():
    print("PyTorch cannot initialize CUDA in this container.", file=sys.stderr)
    print("Restart it with NVIDIA GPU access, then verify: python3 -c 'import torch; print(torch.cuda.is_available())'", file=sys.stderr)
    raise SystemExit(3)
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "${OUTPUT_DIR}"
cd "${MODEL_CODE_ROOT}"

echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Dataset: ${DATA_FILE}"
echo "Images: ${IMAGE_FOLDER}"
echo "Output: ${OUTPUT_DIR}"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Dynamic SAM3 candidate view: ${SAM3_DYNAMIC_CANDIDATE_VIEW}"

# train.py automatically resumes when OUTPUT_DIR contains checkpoint-*.
python3 -m point.train.train \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version v1 \
    --data_path "${DATA_FILE}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${CLIP_VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_sam3_conditioning true \
    --mm_sam3_vision_tower "${SAM3_VISION_TOWER}" \
    --mm_sam3_device vision \
    --mm_sam3_dtype bfloat16 \
    --mm_sam3_candidate_attention true \
    --mm_sam3_candidate_loss_weight 0.2 \
    --tune_sam3_fusion true \
    --tune_mm_mlp_adapter "${TUNE_MM_MLP_ADAPTER}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --mm_use_im_start_end false \
    --mm_use_im_patch_token false \
    --image_aspect_ratio pad \
    "${DYNAMIC_CANDIDATE_ARGS[@]}" \
    --freeze_backbone true \
    --lora_enable true \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout 0.05 \
    --bits 4 \
    --bf16 true \
    --tf32 true \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0.0 \
    --warmup_steps "${WARMUP_STEPS}" \
    --lr_scheduler_type cosine \
    --logging_steps "${LOGGING_STEPS}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing true \
    --dataloader_num_workers "${NUM_WORKERS}" \
    --lazy_preprocess true \
    --group_by_modality_length true \
    --remove_unused_columns false \
    --report_to "${REPORT_TO}"
