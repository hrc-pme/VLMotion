#!/usr/bin/env bash

set -euo pipefail

# 用法:
#   ./docker/run.sh [--root] [容器名稱] [ROS_DOMAIN_ID]
#   ./docker/run.sh [--root] [ROS_DOMAIN_ID]
# 說明:
#   - 若第一個非 --root 的參數是數字 (0-232)，視為 ROS_DOMAIN_ID（容器名稱使用預設）
#   - 若第一個是名稱，第二個是數字，則分別視為 容器名稱 與 ROS_DOMAIN_ID
# 預設容器名稱: stretch3_vla_container

EXEC_USER=""
DOMAIN_ID=""
if [[ "${1-}" == "--root" ]]; then
  EXEC_USER="root"
  shift || true
fi

# 解析參數：支援 --root、容器名稱與數字型 ROS_DOMAIN_ID
CONTAINER_NAME="vlpoint-inference"

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
# 先允許任何用戶連接到X11服務器（臨時解決方案）
xhost +local: 2>/dev/null || true

# 在容器內設置X11認證
docker exec -u root "${CONTAINER_NAME}" bash -c '
    # 創建root用戶的.Xauthority文件
    if [ ! -f /root/.Xauthority ]; then
        touch /root/.Xauthority
        chmod 600 /root/.Xauthority
    fi
    
    # 清空現有的認證並添加新的
    xauth remove :1 2>/dev/null || true
    echo "'"$(xauth list | grep ":1 " | head -1)"'" | xauth merge -
' 2>/dev/null || true

echo "進入容器: ${CONTAINER_NAME}"
if [[ -n "${DOMAIN_ID}" ]]; then
  RUNTIME_DOMAIN_ID="${DOMAIN_ID}"
else
  # 若使用者未指定，沿用容器內已設定值，最後 fallback 到 0
  RUNTIME_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
fi

echo "會在容器內自動 source /workspace/environment.sh（ROS_DOMAIN_ID=${RUNTIME_DOMAIN_ID}）"
if [[ -n "${EXEC_USER}" ]]; then
  exec docker exec -u "${EXEC_USER}" -it "${CONTAINER_NAME}" bash -lc \
    "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh ${RUNTIME_DOMAIN_ID}; else echo '提示: /workspace/environment.sh 不存在，略過'; fi; exec bash -i"
else
  exec docker exec -it "${CONTAINER_NAME}" bash -lc \
    "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh ${RUNTIME_DOMAIN_ID}; else echo '提示: /workspace/environment.sh 不存在，略過'; fi; exec bash -i"
fi
