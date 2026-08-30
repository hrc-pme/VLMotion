#!/usr/bin/env bash

# Set ROS_DOMAIN_ID with validation and source ROS environments

MIN_DOMAIN_ID=0
MAX_DOMAIN_ID=232

ROS_DOMAIN_ID_INPUT=${1:-0}

if [[ "$ROS_DOMAIN_ID_INPUT" =~ ^[0-9]+$ ]] && \
   [[ "$ROS_DOMAIN_ID_INPUT" -ge "$MIN_DOMAIN_ID" ]] && \
   [[ "$ROS_DOMAIN_ID_INPUT" -le "$MAX_DOMAIN_ID" ]]; then
  export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_INPUT"
  echo "ROS_DOMAIN_ID set to $ROS_DOMAIN_ID"
else
  echo "Invalid ROS_DOMAIN_ID. Provide a value between $MIN_DOMAIN_ID and $MAX_DOMAIN_ID."
  # If sourced, 'return', else 'exit'
  case $- in
    *i*) ;; # ignore interactive flag
  esac
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 1 2>/dev/null || true
  else
    exit 1
  fi
fi

# Reduce noisy Python packaging warnings during colcon builds
# - Silence setuptools 'develop' deprecation and related messages
# - Disable pip version check notices
# - Prevent setuptools_scm from probing git (avoids toml/git warnings)
# Suppress noisy packaging warnings
# - message-based filters
export PYTHONWARNINGS="${PYTHONWARNINGS:-}$( [[ -n ${PYTHONWARNINGS:-} ]] && echo , )ignore:develop command is deprecated,ignore:setup.py install is deprecated,ignore:easy_install command is deprecated,ignore:toml section missing"
# 避免使用類別型過濾（在不同 setuptools 版本下容易失敗）
# 僅保留訊息式過濾，避免 "Invalid -W option ignored" 噪音。
export PIP_DISABLE_PIP_VERSION_CHECK=1
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0}"

# Avoid misleading pip virtualenv warnings during builds
unset VIRTUAL_ENV PIP_REQUIRE_VIRTUALENV || true

# Source ROS 2 base environment (required for ament_cmake)
if [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  echo "Sourced /opt/ros/${ROS_DISTRO:-humble}/setup.bash"
else
  echo "/opt/ros/${ROS_DISTRO:-humble}/setup.bash not found. Is ROS installed?"
fi

# Optionally source current workspace overlay if present
if [ -f "/workspace/ros2_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "/workspace/ros2_ws/install/setup.bash"
  echo "Sourced /workspace/ros2_ws/install/setup.bash"
fi

# Optionally source Stretch3 overlay if present
STRETCH3_OVERLAY_PATH="${STRETCH3_OVERLAY_PATH:-/root/stretch3/ament_ws/install/local_setup.bash}"
if [ -f "${STRETCH3_OVERLAY_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${STRETCH3_OVERLAY_PATH}"
  echo "Sourced ${STRETCH3_OVERLAY_PATH}"
fi

# Keep SSH X11 DISPLAY as-is (localhost:10.0) to match the cookie
# Some setups don’t accept switching to 127.0.0.1 due to xauth matching.

# Sanitize prefix paths to remove non-existent entries (avoids AMENT/CMAKE warnings)
sanitize_prefix_var() {
  local name="$1"
  local val="${!name:-}"
  [[ -z "$val" ]] && return 0
  local out="" sep=""
  IFS=':' read -r -a parts <<< "$val"
  for p in "${parts[@]}"; do
    [[ -z "$p" ]] && continue
    if [[ -d "$p" ]]; then
      out+="${sep}${p}"; sep=":"
    fi
  done
  export "$name=$out"
}

sanitize_prefix_var AMENT_PREFIX_PATH
sanitize_prefix_var CMAKE_PREFIX_PATH
sanitize_prefix_var COLCON_PREFIX_PATH
