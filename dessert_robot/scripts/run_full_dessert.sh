#!/usr/bin/env bash
# Run the FULL dessert sequence autonomously: all eight skills in order, each
# executed by run_dessert_skill.sh with the multitask checkpoint (or an
# explicitly provided checkpoint). The operator confirms after every skill.
#
# Usage: ./scripts/run_full_dessert.sh [--checkpoint PATH] [--duration SECONDS] [--dry-run]
#
# This orchestrator only sequences skills and asks for confirmation; it does
# not reset the scene, retry automatically, or run anything in the background.
# Assumes the LeRobot conda/venv environment is already activated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=dessert_common.sh
source "${SCRIPT_DIR}/dessert_common.sh"

RUN_SKILL="${SCRIPT_DIR}/run_dessert_skill.sh"

usage() {
  echo "Usage: $0 [--checkpoint PATH] [--duration SECONDS] [--dry-run]" >&2
  exit 1
}

# ---- argument parsing ---------------------------------------------------------
CHECKPOINT=""
DURATION=20
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint needs a value" >&2; usage; }
                  CHECKPOINT="$2"; shift 2 ;;
    --duration)   [[ $# -ge 2 ]] || { echo "ERROR: --duration needs a value" >&2; usage; }
                  DURATION="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    *)            echo "ERROR: unsupported argument: $1" >&2; usage ;;
  esac
done

if ! is_positive_int "$DURATION"; then
  echo "ERROR: --duration must be a positive integer (seconds), got '$DURATION'" >&2
  exit 1
fi

[[ -x "$RUN_SKILL" ]] || { echo "ERROR: missing or non-executable: $RUN_SKILL" >&2; exit 1; }
require_pcan

# ---- checkpoint resolution (same rules as run_dessert_skill.sh) ------------------
if [[ -n "$CHECKPOINT" ]]; then
  if is_valid_checkpoint "$CHECKPOINT"; then
    :
  elif is_valid_checkpoint "$CHECKPOINT/pretrained_model"; then
    CHECKPOINT="$CHECKPOINT/pretrained_model"
  else
    echo "ERROR: '$CHECKPOINT' is not a valid checkpoint (needs config.json + model.safetensors," >&2
    echo "       either directly or under pretrained_model/)." >&2
    exit 1
  fi
  RESOLVED_CHECKPOINT="$CHECKPOINT"
else
  if ! RESOLVED_CHECKPOINT="$(resolve_checkpoint "$PROJECT_ROOT/checkpoints/final_full_dessert")"; then
    echo "ERROR: no valid multitask checkpoint found under $PROJECT_ROOT/checkpoints/final_full_dessert/checkpoints" >&2
    echo "       Train it first: ./scripts/train_dessert_vla.sh" >&2
    exit 1
  fi
fi

# ---- per-skill command construction -----------------------------------------------
skill_cmd() {
  local skill="$1"
  local cmd=( "$RUN_SKILL" "$skill" --duration "$DURATION" )
  if [[ -n "$CHECKPOINT" ]]; then
    cmd+=( --checkpoint "$CHECKPOINT" )
  else
    cmd+=( --checkpoint-mode full )
  fi
  printf '%s\n' "${cmd[@]}"
}

# ---- print plan -----------------------------------------------------------------------
echo "=============== run_full_dessert plan ======================="
echo "checkpoint:        $RESOLVED_CHECKPOINT"
echo "duration per skill: ${DURATION}s"
echo "dry-run:           $DRY_RUN"
echo "sequence:"
I=0
for SKILL in "${DESSERT_SKILLS[@]}"; do
  I=$(( I + 1 ))
  echo "  ${I}. ${SKILL}: $(task_for_skill "$SKILL")"
done
echo "============================================================="

if [[ "$DRY_RUN" == true ]]; then
  I=0
  for SKILL in "${DESSERT_SKILLS[@]}"; do
    I=$(( I + 1 ))
    mapfile -t CMD < <(skill_cmd "$SKILL")
    printf '[dry-run] step %d: ' "$I"
    print_cmd "${CMD[@]}"
  done
  echo "[dry-run] not executing. In a real run each step runs live and is gated"
  echo "[dry-run] by run_dessert_skill.sh's own RUN confirmation prompt."
  exit 0
fi

echo
echo "*** FULLY AUTONOMOUS SEQUENCE ***"
echo "The robot will attempt all ${#DESSERT_SKILLS[@]} dessert skills IN ORDER, driven only by"
echo "the trained policy (no teleoperation). Between skills you will be asked to"
echo "confirm success before continuing. The scene is NOT reset automatically —"
echo "you are responsible for fixing the scene if a skill partially fails."
echo "Keep the EMERGENCY STOP within reach at all times. Ctrl+C stops everything."
echo
read -r -p "Type START DESSERT (exactly) to begin: " REPLY
if [[ "$REPLY" != "START DESSERT" ]]; then
  echo "Aborted."
  exit 1
fi

# ---- sequential execution ---------------------------------------------------------------
I=0
for SKILL in "${DESSERT_SKILLS[@]}"; do
  I=$(( I + 1 ))
  while true; do
    echo
    echo "=== Skill ${I} of ${#DESSERT_SKILLS[@]}: ${SKILL} ==="
    mapfile -t CMD < <(skill_cmd "$SKILL")
    "${CMD[@]}"

    while true; do
      read -r -p "Skill ${I} (${SKILL}) completed successfully? [y/r/q] " ANSWER
      case "$ANSWER" in
        y|Y) break 2 ;;                       # next skill
        r|R) echo "Retrying ${SKILL}. Restore the scene to its starting state first."; break ;;
        q|Q)
          echo
          echo "Dessert sequence stopped by operator after skill ${I} (${SKILL})."
          echo "Completed skills: $(( I - 1 )) of ${#DESSERT_SKILLS[@]} confirmed successful."
          exit 0
          ;;
        *)   echo "Please answer y (next skill), r (retry this skill), or q (quit)." ;;
      esac
    done
  done
done

echo
echo "All ${#DESSERT_SKILLS[@]} dessert skills confirmed complete. Dessert finished!"
