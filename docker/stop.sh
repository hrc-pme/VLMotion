#!/usr/bin/env bash

set -euo pipefail

# 停止容器並執行 docker compose down
# 用法：
#   ./docker/stop.sh [容器名稱]
# 預設容器名稱: vla-inference

CONTAINER_NAME=${1:-vla-inference}

if ! command -v docker >/dev/null 2>&1; then
  echo "找不到 docker 指令，請先安裝 Docker" >&2
  exit 1
fi

# 若容器存在且正在運行則停止
if docker ps -a --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
    echo "停止容器：${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" >/dev/null || true
  else
    echo "容器已是停止狀態：${CONTAINER_NAME}"
  fi
else
  echo "容器不存在：${CONTAINER_NAME}"
fi

echo "執行 docker compose down（移除由 Compose 建立的容器/網路）..."
docker compose -f docker/docker-compose.yml down
echo "已完成 docker compose down"
