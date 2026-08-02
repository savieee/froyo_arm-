#!/usr/bin/env bash
# Shared definitions for the dessert_robot scripts. Source this file; do not run it.
# Assumes the caller has set: set -euo pipefail, SCRIPT_DIR, PROJECT_ROOT.

# ---- skills -------------------------------------------------------------------
DESSERT_SKILLS=(
  pick_spatula
  scoop_strawberries
  scoop_blueberries
  scoop_yogurt
  place_spatula
  pour_honey
  place_honey
  return_home
)

task_for_skill() {
  case "$1" in
    pick_spatula)       echo "Pick up the spatula." ;;
    scoop_strawberries) echo "Use the spatula to scoop strawberries and place them into the main bowl." ;;
    scoop_blueberries)  echo "Use the spatula to scoop blueberries and place them into the main bowl." ;;
    scoop_yogurt)       echo "Use the spatula to scoop yogurt and place it into the main bowl." ;;
    place_spatula)      echo "Place the spatula back in its original position." ;;
    pour_honey)         echo "Pick up the honey container and pour honey into the main bowl." ;;
    place_honey)        echo "Place the honey container back in its original position." ;;
    return_home)        echo "Move the robot arm back to its home position." ;;
    *)                  return 1 ;;
  esac
}

skills_list() { printf '%s ' "${DESSERT_SKILLS[@]}"; }

# ---- cameras ------------------------------------------------------------------
# NOTE: numeric indices (0/2) are NOT stable on this machine — /dev/video0 is a
# v4l2loopback device that OpenCV cannot open. Use stable /dev/v4l/by-id paths;
# override via env vars when moving to another machine.
FRONT_CAMERA="${FRONT_CAMERA:-/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/v4l/by-id/usb-LuoKe_Technology_Co.__Ltd._LRCP_G720P_SN0001-video-index0}"
CAMERAS="{front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: 640, height: 480, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: ${SIDE_CAMERA}, width: 640, height: 480, fps: 30, fourcc: MJPG}}"

# ---- validation helpers ---------------------------------------------------------
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

require_pcan() {
  if [[ -z "${PCAN_IF:-}" ]]; then
    echo "ERROR: PCAN_IF is not set. On this laptop run: export PCAN_IF=can0" >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: $1 not found on PATH. Activate the lerobot environment first." >&2
    exit 1
  fi
}

# ---- checkpoint helpers ----------------------------------------------------------
# A valid LeRobot 0.4.4 checkpoint is a pretrained_model/ directory containing
# config.json and model.safetensors, produced under:
#   <output_dir>/checkpoints/<step>/pretrained_model/   (+ "last" symlink)
is_valid_checkpoint() {
  [[ -f "$1/config.json" && -f "$1/model.safetensors" ]]
}

# resolve_checkpoint <train_output_dir>  -> echoes pretrained_model path, or fails
resolve_checkpoint() {
  local base="$1/checkpoints"
  if is_valid_checkpoint "${base}/last/pretrained_model"; then
    echo "${base}/last/pretrained_model"
    return 0
  fi
  if [[ -d "$base" ]]; then
    local step_dir
    for step_dir in $(ls -1 "$base" 2>/dev/null | grep -E '^[0-9]+$' | sort -n -r); do
      if is_valid_checkpoint "${base}/${step_dir}/pretrained_model"; then
        echo "${base}/${step_dir}/pretrained_model"
        return 0
      fi
    done
  fi
  return 1
}

# ---- misc ------------------------------------------------------------------------
print_cmd() {
  printf 'command: '
  printf '%q ' "$@"
  printf '\n'
}

# dataset_info <root> <repo_id> [extra dataset_info.py args...]
# Prints a JSON summary; fails if the dataset cannot be opened by LeRobot.
dataset_info() {
  local root="$1" repo_id="$2"
  shift 2
  python "${SCRIPT_DIR}/dataset_info.py" --root "$root" --repo-id "$repo_id" "$@"
}
