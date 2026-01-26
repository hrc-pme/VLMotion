#!/usr/bin/env bash

set -euo pipefail

# 用法:
#   ./docker/run.sh [--root] [容器名稱] [ROS_DOMAIN_ID]
#   ./docker/run.sh [--root] [ROS_DOMAIN_ID]
# 說明:
#   - 若第一個非 --root 的參數是數字 (0-232)，視為 ROS_DOMAIN_ID（容器名稱使用預設）
#   - 若第一個是名稱，第二個是數字，則分別視為 容器名稱 與 ROS_DOMAIN_ID
# 預設容器名稱: vla-inference

EXEC_USER=""
DOMAIN_ID=""
if [[ "${1-}" == "--root" ]]; then
  EXEC_USER="root"
  shift || true
fi

# 解析參數：支援 --root、容器名稱與數字型 ROS_DOMAIN_ID
CONTAINER_NAME="vla-inference"

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ ${#@} -gt 0 ]]; then
  if is_integer "${1-}"; then
    DOMAIN_ID="$1"
    shift || true
  elif [[ -n "${1-}" ]]; then
    CONTAINER_NAME="$1"
    shift || true
    if [[ ${#@} -gt 0 ]] && is_integer "${1-}"; then
      DOMAIN_ID="$1"
      shift || true
    fi
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "找不到 docker 指令，請先安裝 Docker" >&2
  exit 1
fi

# 先確保服務已啟動（若尚未建立/啟動會自動 up -d）
echo "確保 docker compose 服務已啟動（up -d）..."
docker compose -f docker/docker-compose.yml up -d >/dev/null

# 是否存在此容器（包含已停止）
if ! docker ps -a --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  echo "找不到容器：${CONTAINER_NAME}（已嘗試 docker compose up -d）" >&2
  exit 1
fi

# 若容器存在但未運行，先啟動
if ! docker ps --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  echo "容器未在運行，正在啟動：${CONTAINER_NAME}"
  docker start "${CONTAINER_NAME}" >/dev/null
fi

# 設置X11權限
echo "設置 X11 權限..."
# 为Docker创建一个可访问的X11认证文件
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi
# 允许Docker容器连接到X11
xhost +local:docker 2>/dev/null || true
# 将认证文件复制到容器中
docker cp $XAUTH "${CONTAINER_NAME}:/tmp/.docker.xauth" 2>/dev/null || true

echo "進入容器: ${CONTAINER_NAME}"
if [[ -n "${DOMAIN_ID}" ]]; then
  echo "會在容器內套用 ROS_DOMAIN_ID=${DOMAIN_ID}（若 /workspace/environment.sh 存在）"
  if [[ -n "${EXEC_USER}" ]]; then
    exec docker exec -u "${EXEC_USER}" -it "${CONTAINER_NAME}" bash -lc \
      "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh ${DOMAIN_ID}; else echo '提示: /workspace/environment.sh 不存在，略過'; fi; exec bash -i"
  else
    exec docker exec -it "${CONTAINER_NAME}" bash -lc \
      "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh ${DOMAIN_ID}; else echo '提示: /workspace/environment.sh 不存在，略過'; fi; exec bash -i"
  fi
else
  if [[ -n "${EXEC_USER}" ]]; then
    exec docker exec -u "${EXEC_USER}" -it "${CONTAINER_NAME}" bash
  else
    exec docker exec -it "${CONTAINER_NAME}" bash
  fi
fi
