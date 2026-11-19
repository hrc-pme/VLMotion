#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vlmotion}"
MODEL_SIZE="${MODEL_SIZE:-small}"
MODEL_DIR="${MODEL_DIR:-/models/faster-whisper}"

if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is not running." >&2
    echo "Build/start it first with: ./docker/run.sh --build --no-stt 0" >&2
    exit 1
fi

echo "Downloading multilingual faster-whisper ${MODEL_SIZE} into ${MODEL_DIR}..."
docker exec -i "${CONTAINER_NAME}" python3 -c \
    "from faster_whisper import WhisperModel; WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8', download_root='${MODEL_DIR}', local_files_only=False); print('Speech model is ready.')"

echo "The model is stored in the host models/faster-whisper directory."
echo "Restart VLMotion after the download: ./docker/run.sh --no-follow 0"
