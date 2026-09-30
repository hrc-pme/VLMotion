#!/bin/bash
set -e

RET_OK=0
RET_QUIT=1
RET_RETRY=3
RET_BACK=2

check_tui_tool() {
  if command -v dialog >/dev/null 2>&1; then
    echo "dialog"
  elif command -v whiptail >/dev/null 2>&1; then
    echo "whiptail"
  else
    echo "none"
  fi
}

usage() {
  cat << EOF
usage: $0 [profile] [service] [ros_domain_id]

Select a device (profile) first, then a compose service available on that device.

profile:
  4060ti   — GPU host (CUDA 12.1)
  stretch3 — robot / CPU

services by profile:
  4060ti:   zenoh-router | dev | cb | vl | vlmotion | build | stop
  stretch3: dev | cb | bridge | build | stop

ros_domain_id:
  optional for zenoh-router/dev/vl/vlmotion/bridge; default 0 (0-232)

examples:
  $0                          # TUI: device → service
  $0 4060ti                   # TUI: services for 4060ti
  $0 4060ti zenoh-router 30   # sole Zenoh router with ROS_DOMAIN_ID=30
  $0 4060ti vl 30             # GUI, base held (auto-starts zenoh-router)
  $0 4060ti vlmotion 30       # GUI, base moves
  $0 stretch3 bridge 30       # cmd_vel gate (needs hellorobot cams + host router)
  $0 stretch3 stop
  VLMOTION_PROFILE=4060ti $0 vl 30
EOF
  exit 1
}

show_error() {
  local tui_tool=$1
  local message=$2

  if [ "$tui_tool" = "dialog" ]; then
    dialog --title "Error" --msgbox "$message" 10 60
    clear
  elif [ "$tui_tool" = "whiptail" ]; then
    whiptail --title "Error" --msgbox "$message" 10 60
  else
    echo "Error: $message" >&2
  fi
  exit 1
}

auto_detect_profile() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "4060ti"
  else
    echo "stretch3"
  fi
}

is_profile() {
  case "$1" in
    stretch3|4060ti) return 0 ;;
    *) return 1 ;;
  esac
}

validate_profile() {
  is_profile "$1" || usage
}

# Space-separated list of services allowed for a profile.
services_for_profile() {
  case "$1" in
    4060ti)   echo "zenoh-router dev cb vl vlmotion build stop" ;;
    stretch3) echo "dev cb bridge build stop" ;;
    *) return 1 ;;
  esac
}

service_allowed_on_profile() {
  local profile=$1
  local service=$2
  local allowed s
  allowed=$(services_for_profile "$profile")
  for s in $allowed; do
    [ "$s" = "$service" ] && return 0
  done
  return 1
}

validate_service_for_profile() {
  local profile=$1
  local service=$2
  if ! service_allowed_on_profile "$profile" "$service"; then
    echo "Error: service '$service' is not available on profile '$profile'." >&2
    echo "Available: $(services_for_profile "$profile")" >&2
    exit 1
  fi
}

resolve_profile() {
  local explicit_profile=${1:-}

  if [ -n "$explicit_profile" ]; then
    validate_profile "$explicit_profile"
    echo "$explicit_profile"
    return
  fi

  if [ -n "${VLMOTION_PROFILE:-}" ]; then
    validate_profile "$VLMOTION_PROFILE"
    echo "$VLMOTION_PROFILE"
    return
  fi

  auto_detect_profile
}

resolve_compose_file() {
  local script_dir=$1
  local profile=$2

  validate_profile "$profile"
  case "$profile" in
    stretch3) echo "$script_dir/docker/stretch3.compose.yaml" ;;
    4060ti)   echo "$script_dir/docker/4060ti.compose.yaml" ;;
  esac
}

# Run dialog/whiptail and capture selection + exit status under `set -e`.
# Sets globals: __tui_selection, __tui_exit_code
tui_run_dialog() {
  __tui_selection=""
  __tui_exit_code=0
  __tui_selection=$(DIALOGRC=/dev/null dialog "$@" 2>&1 >/dev/tty) || __tui_exit_code=$?
}

tui_run_whiptail() {
  __tui_selection=""
  __tui_exit_code=0
  __tui_selection=$(whiptail "$@" 3>&1 1>&2 2>&3) || __tui_exit_code=$?
}

select_profile_tui() {
  local tui_tool=$1
  local default_profile=${2:-$(auto_detect_profile)}
  local cancel_label=${3:-Quit}
  local menu_options=(
    4060ti "| GPU host / CUDA 12.1"
    stretch3 "| Robot / CPU (stretch3)"
  )

  if [ "$tui_tool" = "dialog" ]; then
    tui_run_dialog \
      --backtitle "VLMotion Docker Manager" \
      --title " Select Device " \
      --default-item "$default_profile" \
      --cancel-label "$cancel_label" \
      --menu "Choose device (profile):" 15 70 2 \
      "${menu_options[@]}"
    clear
  else
    tui_run_whiptail \
      --backtitle "VLMotion Docker Manager" \
      --title " Select Device " \
      --default-item "$default_profile" \
      --cancel-button "$cancel_label" \
      --menu "Choose device (profile):" 15 70 2 \
      "${menu_options[@]}"
  fi

  [ "$__tui_exit_code" -ne 0 ] && return "$RET_QUIT"
  [ -z "$__tui_selection" ] && return "$RET_RETRY"
  echo "$__tui_selection"
  return "$RET_OK"
}

select_service_tui() {
  local tui_tool=$1
  local profile=$2
  local -a menu_options=()
  local n_items=0

  case "$profile" in
    4060ti)
      menu_options=(
        zenoh-router "| Zenoh router"
        dev "| Dev shell"
        cb "| Colcon build"
        vl "| GUI, base held"
        vlmotion "| GUI, base moves"
        build "| Rebuild image"
        stop "| Stop all"
      )
      n_items=7
      ;;
    stretch3)
      menu_options=(
        dev "| Dev shell"
        cb "| Colcon build"
        bridge "| Robot bridge"
        build "| Rebuild image"
        stop "| Stop all"
      )
      n_items=5
      ;;
    *)
      echo "Error: unknown profile: $profile" >&2
      return "$RET_QUIT"
      ;;
  esac

  if [ "$tui_tool" = "dialog" ]; then
    tui_run_dialog \
      --backtitle "VLMotion Docker Manager" \
      --title " Select Service ($profile) " \
      --cancel-label "Back" \
      --menu "Choose a service for $profile:" $((12 + n_items)) 72 "$n_items" \
      "${menu_options[@]}"
    clear
  else
    tui_run_whiptail \
      --backtitle "VLMotion Docker Manager" \
      --title " Select Service ($profile) " \
      --cancel-button "Back" \
      --menu "Choose a service for $profile:" $((12 + n_items)) 72 "$n_items" \
      "${menu_options[@]}"
  fi

  [ "$__tui_exit_code" -ne 0 ] && return "$RET_BACK"
  [ -z "$__tui_selection" ] && return "$RET_RETRY"
  echo "$__tui_selection"
  return "$RET_OK"
}

select_ros_domain_id_tui() {
  local tui_tool=$1
  local default_id=${2:-0}

  if [ "$tui_tool" = "dialog" ]; then
    tui_run_dialog \
      --backtitle "VLMotion Docker Manager" \
      --title " ROS Domain ID " \
      --cancel-label "Back" \
      --inputbox "Enter ROS Domain ID (0-232):" 10 50 "$default_id"
    clear
  else
    tui_run_whiptail \
      --backtitle "VLMotion Docker Manager" \
      --title " ROS Domain ID " \
      --cancel-button "Back" \
      --inputbox "Enter ROS Domain ID (0-232):" 10 50 "$default_id"
  fi

  [ "$__tui_exit_code" -ne 0 ] && return "$RET_BACK"
  echo "$__tui_selection"
  return "$RET_OK"
}

# Capture stdout + exit code without tripping `set -e` on Back/Cancel.
tui_ask() {
  # usage: tui_ask <out_var> <rc_var> <command> [args...]
  local __out_var=$1
  local __rc_var=$2
  shift 2
  local __out __rc=0
  __out=$("$@") || __rc=$?
  printf -v "$__out_var" '%s' "$__out"
  printf -v "$__rc_var" '%s' "$__rc"
}

validate_ros_domain_id() {
  [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 0 ] && [ "$1" -le 232 ] || usage
}

# Compose args: always resolve root .env for ${VAR} interpolation + container env_file.
compose_cmd() {
  local project_name=$1
  local compose_file=$2
  shift 2

  local -a args=(-p "$project_name" -f "$compose_file")
  local env_file
  env_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
  if [ -f "$env_file" ]; then
    args+=(--env-file "$env_file")
  fi
  docker compose "${args[@]}" "$@"
}

# Remove existing container(s) for service(s), then bring them up fresh
# so env changes (e.g. ROS_DOMAIN_ID) always take effect.
compose_recreate() {
  local project_name=$1
  local compose_file=$2
  shift 2

  if [ "$#" -eq 0 ]; then
    echo "compose_recreate: need at least one service" >&2
    return 1
  fi
  echo "[run.sh] recreate: $*"
  compose_cmd "$project_name" "$compose_file" rm -sf "$@" >/dev/null 2>&1 || true
  compose_cmd "$project_name" "$compose_file" up -d "$@"
}

wait_zenoh_router() {
  local port="${ZENOH_ROUTER_PORT:-7447}"
  local i

  for i in $(seq 1 40); do
    if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      echo "[run.sh] zenoh-router ready on :${port}"
      return 0
    fi
    sleep 0.25
  done
  echo "[run.sh] warning: zenoh-router not accepting :${port} yet (check: docker logs vlmotion-4060ti-zenoh-router)" >&2
  return 0
}

# Host-only: recreate zenoh-router and wait until :7447 accepts connections.
ensure_zenoh_router() {
  local project_name=$1
  local compose_file=$2

  if [[ "$compose_file" != *4060ti* ]]; then
    return 0
  fi
  compose_recreate "$project_name" "$compose_file" zenoh-router
  wait_zenoh_router
}

start_dev() {
  local project_name=$1
  local compose_file=$2

  ensure_zenoh_router "$project_name" "$compose_file"
  compose_recreate "$project_name" "$compose_file" dev
  compose_cmd "$project_name" "$compose_file" exec dev /bin/bash
}

# Allow Docker containers to open Host Dear PyGui / X11 windows.
ensure_x11_for_docker() {
  if [ -z "${DISPLAY:-}" ]; then
    echo "[run.sh] warning: DISPLAY is empty; Host GUI may fail" >&2
    return 0
  fi
  if ! command -v xhost >/dev/null 2>&1; then
    echo "[run.sh] warning: xhost not found; cannot authorize Docker for X11" >&2
    return 0
  fi
  if xhost +local:docker >/dev/null 2>&1; then
    echo "[run.sh] X11: xhost +local:docker (DISPLAY=${DISPLAY})"
  else
    echo "[run.sh] warning: xhost +local:docker failed; Host GUI may fail to open display" >&2
  fi
}

start_host_gui() {
  local project_name=$1
  local compose_file=$2
  local service=$3

  ensure_x11_for_docker
  ensure_zenoh_router "$project_name" "$compose_file"
  compose_recreate "$project_name" "$compose_file" "$service"
  echo "[run.sh] ${service} running in background (container: vlmotion-4060ti-${service})"
  echo "[run.sh] logs: docker logs -f vlmotion-4060ti-${service}"
  echo "[run.sh] stop:  ./run.sh 4060ti stop"
}

start_bridge() {
  local project_name=$1
  local compose_file=$2

  # Daemon only — no interactive shell. robot_bridge runs as container PID 1.
  compose_recreate "$project_name" "$compose_file" bridge
  echo "[run.sh] bridge running in background (container: vlmotion-stretch3-bridge)"
  echo "[run.sh] logs: docker logs -f vlmotion-stretch3-bridge"
  echo "[run.sh] stop:  ./run.sh stretch3 stop"
}

start_zenoh_router() {
  local project_name=$1
  local compose_file=$2

  compose_recreate "$project_name" "$compose_file" zenoh-router
  wait_zenoh_router
  echo "zenoh-router is running (container: vlmotion-4060ti-zenoh-router, ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0})"
}

build_image() {
  local project_name=$1
  local compose_file=$2

  if [[ "$compose_file" == *4060ti* ]]; then
    compose_cmd "$project_name" "$compose_file" build dev zenoh-router
  else
    compose_cmd "$project_name" "$compose_file" build dev
  fi
}

colcon_build() {
  local project_name=$1
  local compose_file=$2

  # cb entrypoint needs the router (client wait).
  ensure_zenoh_router "$project_name" "$compose_file"
  echo "[run.sh] recreate: cb"
  compose_cmd "$project_name" "$compose_file" rm -sf cb >/dev/null 2>&1 || true
  compose_cmd "$project_name" "$compose_file" run --rm cb
}

stop_services() {
  local project_name=$1
  local compose_file=$2

  compose_cmd "$project_name" "$compose_file" down
}

ensure_compose_file() {
  local script_dir=$1
  local profile=$2

  compose_file=$(resolve_compose_file "$script_dir" "$profile")
  if [ ! -f "$compose_file" ]; then
    echo "Error: compose file not found: $compose_file" >&2
    exit 1
  fi
  echo "Using profile: $profile"
}

needs_ros_domain_id() {
  case "$1" in
    zenoh-router|dev|vl|vlmotion|bridge) return 0 ;;
    *) return 1 ;;
  esac
}

run_service() {
  local project_name=$1
  local compose_file=$2
  local profile=$3
  local service=$4
  local ros_domain_id=${5:-0}

  validate_service_for_profile "$profile" "$service"

  case "$service" in
    zenoh-router)
      validate_ros_domain_id "$ros_domain_id"
      export ROS_DOMAIN_ID=$ros_domain_id
      start_zenoh_router "$project_name" "$compose_file"
      ;;
    dev)
      validate_ros_domain_id "$ros_domain_id"
      export ROS_DOMAIN_ID=$ros_domain_id
      start_dev "$project_name" "$compose_file"
      ;;
    vl|vlmotion)
      validate_ros_domain_id "$ros_domain_id"
      export ROS_DOMAIN_ID=$ros_domain_id
      start_host_gui "$project_name" "$compose_file" "$service"
      ;;
    bridge)
      validate_ros_domain_id "$ros_domain_id"
      export ROS_DOMAIN_ID=$ros_domain_id
      start_bridge "$project_name" "$compose_file"
      ;;
    cb)
      colcon_build "$project_name" "$compose_file"
      ;;
    build)
      build_image "$project_name" "$compose_file"
      ;;
    stop)
      stop_services "$project_name" "$compose_file"
      ;;
    *)
      usage
      ;;
  esac
}

run_tui_for_profile() {
  local tui_tool=$1
  local script_dir=$2
  local project_name=$3
  local profile=$4
  local service ros_domain_id rc

  ensure_compose_file "$script_dir" "$profile"

  while true; do
    # Service menu — Back returns to device selection.
    tui_ask service rc select_service_tui "$tui_tool" "$profile"
    [ "$rc" -eq "$RET_BACK" ] && return "$RET_BACK"
    [ "$rc" -eq "$RET_RETRY" ] && continue
    [ "$rc" -ne "$RET_OK" ] && return "$rc"

    ros_domain_id=0
    if needs_ros_domain_id "$service"; then
      # Domain ID — Back returns to service menu (not exit).
      tui_ask ros_domain_id rc select_ros_domain_id_tui "$tui_tool" 0
      [ "$rc" -ne "$RET_OK" ] && continue
      validate_ros_domain_id "$ros_domain_id"
    fi

    run_service "$project_name" "$compose_file" "$profile" "$service" "$ros_domain_id"

    # Interactive shells leave the manager; daemon / one-shots return to menu.
    case "$service" in
      dev) return "$RET_OK" ;;
      zenoh-router|vl|vlmotion|bridge|cb|build|stop)
        read -r -p "Press Enter to continue..."
        ;;
      *)
        read -r -p "Press Enter to continue..."
        ;;
    esac
  done
}

run_interactive_tui() {
  local tui_tool=$1
  local script_dir=$2
  local project_name=$3
  local initial_profile=${4:-}
  local profile rc

  while true; do
    if [ -n "$initial_profile" ]; then
      profile=$initial_profile
      initial_profile=""
    else
      tui_ask profile rc select_profile_tui "$tui_tool" "$(auto_detect_profile)" "Quit"
      [ "$rc" -eq "$RET_QUIT" ] && return "$RET_OK"
      [ "$rc" -eq "$RET_RETRY" ] && continue
      [ "$rc" -ne "$RET_OK" ] && continue
    fi

    # Must absorb non-zero (Back) under set -e, or the whole script exits.
    rc=0
    run_tui_for_profile "$tui_tool" "$script_dir" "$project_name" "$profile" || rc=$?
    # Back from service menu → device menu again.
    [ "$rc" -eq "$RET_BACK" ] && continue
    # Finished interactive service (dev) or other completion.
    [ "$rc" -eq "$RET_OK" ] && return "$RET_OK"
    continue
  done
}

main() {
  local script_dir compose_file project_name service ros_domain_id tui_tool rc profile
  local arg1 arg2 arg3

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_name="vlmotion"

  # ---- Interactive: device → service -----------------------------------------
  if [ $# -eq 0 ]; then
    tui_tool=$(check_tui_tool)
    [ "$tui_tool" = "none" ] && usage
    run_interactive_tui "$tui_tool" "$script_dir" "$project_name"
    exit 0
  fi

  # ---- CLI: profile-first ----------------------------------------------------
  # Forms:
  #   ./run.sh <profile> [service] [ros_domain_id]
  #   ./run.sh <service> ...   (profile from VLMOTION_PROFILE or auto-detect; service must be allowed)
  arg1=$1
  arg2=${2:-}
  arg3=${3:-}

  if is_profile "$arg1"; then
    profile=$arg1
    service=$arg2
    ros_domain_id=${arg3:-0}

    if [ -z "$service" ]; then
      tui_tool=$(check_tui_tool)
      if [ "$tui_tool" = "none" ]; then
        echo "Error: service required. Available on $profile: $(services_for_profile "$profile")" >&2
        exit 1
      fi
      # Start on this device's service menu; Back goes to device picker (not exit).
      run_interactive_tui "$tui_tool" "$script_dir" "$project_name" "$profile"
      exit 0
    fi
  else
    # Backward-compatible: ./run.sh <service> [ros_domain_id|profile] ...
    service=$arg1
    if [ -n "$arg2" ] && is_profile "$arg2"; then
      profile=$arg2
      ros_domain_id=${arg3:-0}
    elif [ -n "$arg2" ] && [[ "$arg2" =~ ^[0-9]+$ ]]; then
      profile=$(resolve_profile)
      ros_domain_id=$arg2
    else
      profile=$(resolve_profile "${arg2:-}")
      ros_domain_id=${arg3:-0}
    fi
  fi

  validate_profile "$profile"
  validate_service_for_profile "$profile" "$service"
  ensure_compose_file "$script_dir" "$profile"
  run_service "$project_name" "$compose_file" "$profile" "$service" "$ros_domain_id"
}

main "$@"
