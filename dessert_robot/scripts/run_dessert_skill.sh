#!/usr/bin/env bash
# Run ONE dessert skill AUTONOMOUSLY on the follower arm, using either a
# single-skill checkpoint or the full multitask checkpoint.
# No leader arm / teleoperation is involved: the policy drives the robot.
#
# Usage: ./scripts/run_dessert_skill.sh <skill> [--checkpoint PATH] [--checkpoint-mode single|full] [--duration SECONDS] [--dry-run]
#
# Assumes the LeRobot conda/venv environment is already activated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=dessert_common.sh
source "${SCRIPT_DIR}/dessert_common.sh"

usage() {
  echo "Usage: $0 <skill> [--checkpoint PATH] [--checkpoint-mode single|full] [--duration SECONDS] [--dry-run]" >&2
  echo "  skills: $(skills_list)" >&2
  exit 1
}

# ---- argument parsing ---------------------------------------------------------
SKILL=""
CHECKPOINT=""
CHECKPOINT_MODE="full"
DURATION=20
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint needs a value" >&2; usage; }
                       CHECKPOINT="$2"; shift 2 ;;
    --checkpoint-mode) [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint-mode needs a value" >&2; usage; }
                       CHECKPOINT_MODE="$2"; shift 2 ;;
    --duration)        [[ $# -ge 2 ]] || { echo "ERROR: --duration needs a value" >&2; usage; }
                       DURATION="$2"; shift 2 ;;
    --dry-run)         DRY_RUN=true; shift ;;
    --*)               echo "ERROR: unsupported argument: $1" >&2; usage ;;
    *)
      if [[ -z "$SKILL" ]]; then SKILL="$1"; shift
      else echo "ERROR: unexpected extra argument: $1" >&2; usage
      fi
      ;;
  esac
done

[[ -n "$SKILL" ]] || usage

if ! TASK="$(task_for_skill "$SKILL")"; then
  echo "ERROR: unsupported skill '$SKILL' (expected: $(skills_list))" >&2
  exit 1
fi

if [[ "$CHECKPOINT_MODE" != "single" && "$CHECKPOINT_MODE" != "full" ]]; then
  echo "ERROR: --checkpoint-mode must be 'single' or 'full', got '$CHECKPOINT_MODE'" >&2
  exit 1
fi

if ! is_positive_int "$DURATION"; then
  echo "ERROR: --duration must be a positive integer (seconds), got '$DURATION'" >&2
  exit 1
fi

require_pcan
require_cmd lerobot-record

# ---- checkpoint resolution ------------------------------------------------------
if [[ -n "$CHECKPOINT" ]]; then
  # Explicit path: accept the pretrained_model dir itself or a dir containing it.
  if is_valid_checkpoint "$CHECKPOINT"; then
    :
  elif is_valid_checkpoint "$CHECKPOINT/pretrained_model"; then
    CHECKPOINT="$CHECKPOINT/pretrained_model"
  else
    echo "ERROR: '$CHECKPOINT' is not a valid checkpoint (needs config.json + model.safetensors," >&2
    echo "       either directly or under pretrained_model/)." >&2
    exit 1
  fi
else
  if [[ "$CHECKPOINT_MODE" == "single" ]]; then
    TRAIN_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$SKILL"
    TRAIN_HINT="./scripts/train_dessert_skill.sh $SKILL"
  else
    TRAIN_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/final_full_dessert"
    TRAIN_HINT="./scripts/train_dessert_vla.sh"
  fi
  if ! CHECKPOINT="$(resolve_checkpoint "$TRAIN_OUTPUT_DIR")"; then
    echo "ERROR: no valid checkpoint found under $TRAIN_OUTPUT_DIR/checkpoints" >&2
    echo "       Train first: $TRAIN_HINT" >&2
    exit 1
  fi
fi

# ---- camera key mapping for inference ------------------------------------------------
# The checkpoint was trained expecting observation.images.camera1/camera2 (mapped
# from front/side with --rename_map at training time). The live robot still
# produces front/side, so lerobot-record needs the same mapping:
#  - --dataset.rename_map renames the observation keys (and stats) fed to the
#    policy's preprocessor.
#  - lerobot-record 0.4.4 additionally validates camera keys between the new
#    eval dataset (front/side) and the policy config (camera1/camera2) WITHOUT
#    applying rename_map (make_policy is called without it), so the checkpoint's
#    declared inputs must also be extended with the robot keys for that check to
#    pass. Only key membership matters; the shapes are copied from the checkpoint
#    and these extra declared keys are never fed to the model (after renaming,
#    the batch contains camera1/camera2 only, and absent keys are skipped).
RENAME_MAP_JSON='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'

POLICY_EXTRA_INPUTS_JSON="$(python - "$CHECKPOINT" <<'PYEOF'
import json
import sys

with open(f"{sys.argv[1]}/config.json") as f:
    cfg = json.load(f)
feats = cfg.get("input_features", {})
for key in ("observation.images.camera1", "observation.images.camera2"):
    if key not in feats:
        print(f"ERROR: checkpoint config does not declare {key}; "
              "was it trained with the front->camera1 / side->camera2 mapping?",
              file=sys.stderr)
        raise SystemExit(1)
print(json.dumps({
    "observation.images.front": {"type": "VISUAL",
                                 "shape": list(feats["observation.images.camera1"]["shape"])},
    "observation.images.side": {"type": "VISUAL",
                                "shape": list(feats["observation.images.camera2"]["shape"])},
}))
PYEOF
)" || exit 1

# ---- evaluation dataset locations --------------------------------------------------
# Each run records into a fresh timestamped directory under EVALUATION_ROOT:
# LeRobot refuses to record into an existing dataset directory, and this keeps
# retries and repeat evaluations from colliding.
EVALUATION_ROOT="$PROJECT_ROOT/evaluations/$CHECKPOINT_MODE/$SKILL"
EVALUATION_REPO_ID="seeed_rebot_b601_rs/eval_${CHECKPOINT_MODE}_dessert_$SKILL"
RUN_ROOT="${EVALUATION_ROOT}/$(date +%Y%m%d_%H%M%S)"
if [[ "$DRY_RUN" != true ]]; then
  mkdir -p "$EVALUATION_ROOT"
fi

# ---- build command --------------------------------------------------------------------
CMD=(
  lerobot-record
  --robot.type=seeed_b601_rs_follower
  --robot.port="$PCAN_IF"
  --robot.id=follower1
  --robot.can_adapter=socketcan
  --robot.cameras="$CAMERAS"
  --display_data=true
  --policy.path="$CHECKPOINT"
  --policy.device=cuda
  --policy.input_features="$POLICY_EXTRA_INPUTS_JSON"
  --dataset.rename_map="$RENAME_MAP_JSON"
  --dataset.repo_id="$EVALUATION_REPO_ID"
  --dataset.root="$RUN_ROOT"
  --dataset.num_episodes=1
  --dataset.single_task="$TASK"
  --dataset.push_to_hub=false
  --dataset.episode_time_s="$DURATION"
  --dataset.reset_time_s=1
)

# ---- print resolved configuration --------------------------------------------------------
echo "============= run_dessert_skill configuration =============="
echo "skill:           $SKILL"
echo "task:            $TASK"
echo "checkpoint mode: $CHECKPOINT_MODE"
echo "checkpoint:      $CHECKPOINT"
echo "evaluation root: $EVALUATION_ROOT"
echo "this run:        $RUN_ROOT"
echo "eval repo id:    $EVALUATION_REPO_ID"
echo "duration:        ${DURATION}s (single episode, no looping, no retry)"
echo "PCAN interface:  $PCAN_IF"
echo "front camera:    $FRONT_CAMERA"
echo "side camera:     $SIDE_CAMERA"
echo "camera mapping:  front -> camera1, side -> camera2 (policy input keys)"
echo "policy device:   cuda"
echo "dry-run:         $DRY_RUN"
echo "============================================================="
print_cmd "${CMD[@]}"

if [[ "$DRY_RUN" == true ]]; then
  echo "[dry-run] not executing."
  exit 0
fi

echo
echo "*** AUTONOMOUS MODE ***"
echo "The trained policy will drive the robot arm ON ITS OWN for ${DURATION} seconds"
echo "to perform: $TASK"
echo "No teleoperation is active. Keep the EMERGENCY STOP within reach and stay"
echo "clear of the arm's workspace. Ctrl+C stops the process."
echo
read -r -p "Type RUN (exactly) to start autonomous execution: " REPLY
if [[ "$REPLY" != "RUN" ]]; then
  echo "Aborted."
  exit 1
fi

"${CMD[@]}"
