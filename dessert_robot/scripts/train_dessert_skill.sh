#!/usr/bin/env bash
# Fine-tune SmolVLA on ONE dessert skill dataset (independent single-skill test).
#
# Usage: ./scripts/train_dessert_skill.sh <skill> [--steps N] [--batch-size N] [--save-freq N] [--resume] [--dry-run]
#
# Assumes the LeRobot conda/venv environment is already activated.
# Batch size defaults to 1 as a conservative choice for ~7.63 GB VRAM;
# it is a starting point, not a tuned hyperparameter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=dessert_common.sh
source "${SCRIPT_DIR}/dessert_common.sh"

PRETRAINED_MODEL="lerobot/smolvla_base"

# The dataset records cameras as observation.images.front/side, but smolvla_base
# expects observation.images.camera1/camera2. The mapping is applied at training
# time only (--rename_map renames batch keys via the preprocessor); dataset files
# are never touched.
RENAME_MAP_JSON='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'

usage() {
  echo "Usage: $0 <skill> [--steps N] [--batch-size N] [--save-freq N] [--resume] [--dry-run]" >&2
  echo "  skills: $(skills_list)" >&2
  exit 1
}

# ---- argument parsing ---------------------------------------------------------
SKILL=""
STEPS=20000
BATCH_SIZE=1
SAVE_FREQ=2000
RESUME=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)      [[ $# -ge 2 ]] || { echo "ERROR: --steps needs a value" >&2; usage; }
                  STEPS="$2"; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || { echo "ERROR: --batch-size needs a value" >&2; usage; }
                  BATCH_SIZE="$2"; shift 2 ;;
    --save-freq)  [[ $# -ge 2 ]] || { echo "ERROR: --save-freq needs a value" >&2; usage; }
                  SAVE_FREQ="$2"; shift 2 ;;
    --resume)     RESUME=true; shift ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --*)          echo "ERROR: unsupported argument: $1" >&2; usage ;;
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

is_positive_int "$STEPS"      || { echo "ERROR: --steps must be a positive integer, got '$STEPS'" >&2; exit 1; }
is_positive_int "$BATCH_SIZE" || { echo "ERROR: --batch-size must be a positive integer, got '$BATCH_SIZE'" >&2; exit 1; }
is_positive_int "$SAVE_FREQ"  || { echo "ERROR: --save-freq must be a positive integer, got '$SAVE_FREQ'" >&2; exit 1; }

# ---- locations ------------------------------------------------------------------
DATASET_ROOT="$PROJECT_ROOT/datasets/$SKILL"
DATASET_REPO_ID="seeed_rebot_b601_rs/dessert_$SKILL"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$SKILL"
JOB_NAME="dessert_$SKILL"
RESUME_TRAIN_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
# Local 2-camera derivative of smolvla_base, regenerated before each fresh run
# from the HF cache + dataset metadata (kept outside OUTPUT_DIR: lerobot-train
# refuses a pre-existing output directory).
BASE_MODEL_DIR="$PROJECT_ROOT/models/smolvla_base_2cam"

# ---- validation --------------------------------------------------------------------
require_cmd lerobot-train
require_cmd python

if [[ ! -d "$DATASET_ROOT" ]] || [[ -z "$(ls -A "$DATASET_ROOT" 2>/dev/null)" ]]; then
  echo "ERROR: dataset directory missing or empty: $DATASET_ROOT" >&2
  echo "       Record it first: ./scripts/record_dessert_skill.sh $SKILL <num_episodes>" >&2
  exit 1
fi

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "ERROR: CUDA is not available in the active Python environment." >&2
  exit 1
fi

if ! python -c "import lerobot.policies.smolvla.modeling_smolvla" >/dev/null 2>&1; then
  echo "ERROR: SmolVLA is not importable in the active Python environment." >&2
  exit 1
fi

# Opens the dataset with the official LeRobot metadata API; verifies front/side
# cameras, observation.state, action and task labels; prints the summary.
echo "Validating dataset with LeRobot..."
INFO_JSON="$(dataset_info "$DATASET_ROOT" "$DATASET_REPO_ID" --require-features --require-tasks "$TASK")" || {
  echo "ERROR: dataset failed validation (see errors above)." >&2
  exit 1
}
TOTAL_EPISODES="$(printf '%s' "$INFO_JSON" | python -c "import json,sys; print(json.load(sys.stdin)['total_episodes'])")"
TOTAL_FRAMES="$(printf '%s' "$INFO_JSON" | python -c "import json,sys; print(json.load(sys.stdin)['total_frames'])")"

# The front->camera1 / side->camera2 mapping below is only correct if these are
# the dataset's ONLY cameras; an unexpected extra camera would go unmapped.
printf '%s' "$INFO_JSON" | python -c "
import json, sys
keys = sorted(json.load(sys.stdin)['camera_keys'])
expected = ['observation.images.front', 'observation.images.side']
if keys != expected:
    print(f'ERROR: dataset camera keys {keys} != expected {expected}', file=sys.stderr)
    raise SystemExit(1)
"

if [[ "$RESUME" == true ]]; then
  if [[ ! -f "$RESUME_TRAIN_CONFIG" ]]; then
    echo "ERROR: --resume requested but no resumable checkpoint found at:" >&2
    echo "       $RESUME_TRAIN_CONFIG" >&2
    exit 1
  fi
elif [[ -d "${OUTPUT_DIR}/checkpoints" ]]; then
  echo "ERROR: training output already exists: ${OUTPUT_DIR}/checkpoints" >&2
  echo "       Use --resume to continue it, or move/delete the directory." >&2
  exit 1
fi

# ---- 2-camera base model (fresh runs only) ----------------------------------------------
# smolvla_base declares observation.images.camera1/2/3; this robot has two cameras
# and the policy must expect EXACTLY camera1 + camera2 + observation.state. A CLI
# --policy.input_features override cannot achieve that (draccus deep-merges dict
# overrides into the pretrained config.json, so camera3 can never be removed).
# Instead we materialize a local copy of smolvla_base from the HF cache with a
# corrected config (camera3 dropped; image/state/action shapes taken from the
# dataset metadata) and processor pipelines rebuilt with LeRobot's own factory
# (make_pre_post_processors), which is the same code the hub checkpoint was saved
# with. The 906 MB weights are symlinked from the cache, not copied. No network.
if [[ "$RESUME" != true ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] would (re)generate the 2-camera base model at: $BASE_MODEL_DIR"
  else
    echo "Generating 2-camera SmolVLA base model (local HF cache only, no downloads)..."
    DATASET_ROOT="$DATASET_ROOT" DATASET_REPO_ID="$DATASET_REPO_ID" \
    BASE_MODEL_DIR="$BASE_MODEL_DIR" python - <<'PYEOF'
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # generation must never touch the network

dataset_root = os.environ["DATASET_ROOT"]
dataset_repo_id = os.environ["DATASET_REPO_ID"]
base_dir = Path(os.environ["BASE_MODEL_DIR"])
marker = base_dir / "GENERATED_BY_DESSERT_ROBOT"

if base_dir.exists() and any(base_dir.iterdir()) and not marker.exists():
    print(f"ERROR: {base_dir} exists but was not generated by this script; refusing to overwrite.",
          file=sys.stderr)
    raise SystemExit(1)

from huggingface_hub import snapshot_download

snap = Path(snapshot_download("lerobot/smolvla_base", local_files_only=True))

import lerobot.policies  # registers policy config classes for draccus
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors

cfg = PreTrainedConfig.from_pretrained(str(snap))
meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
feats = dataset_to_policy_features(meta.features)

cfg.input_features = {
    "observation.state": feats["observation.state"],
    "observation.images.camera1": feats["observation.images.front"],
    "observation.images.camera2": feats["observation.images.side"],
}
cfg.output_features = {"action": feats["action"]}
cfg.push_to_hub = False

base_dir.mkdir(parents=True, exist_ok=True)
cfg.save_pretrained(str(base_dir))
pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
pre.save_pretrained(str(base_dir))
post.save_pretrained(str(base_dir))

weights = base_dir / "model.safetensors"
if weights.is_symlink() or weights.exists():
    weights.unlink()
weights.symlink_to((snap / "model.safetensors").resolve())
marker.write_text(
    "Generated by the dessert_robot train scripts from the locally cached "
    "lerobot/smolvla_base.\nSafe to delete; regenerated before each fresh training run.\n"
)

for key, ft in cfg.input_features.items():
    print(f"  policy input : {key}  {tuple(ft.shape)}")
for key, ft in cfg.output_features.items():
    print(f"  policy output: {key}  {tuple(ft.shape)}")
PYEOF
  fi
fi

# ---- build command --------------------------------------------------------------------
# Training reads frames keyed front/side; --rename_map renames them to
# camera1/camera2 in the preprocessor before normalization and the model.
if [[ "$RESUME" == true ]]; then
  # LeRobot 0.4.4 resume: all other settings (including rename_map and the
  # policy features) are restored from train_config.json.
  CMD=(
    lerobot-train
    --resume=true
    --config_path="$RESUME_TRAIN_CONFIG"
  )
else
  CMD=(
    lerobot-train
    --policy.path="$BASE_MODEL_DIR"
    --policy.push_to_hub=false
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$DATASET_ROOT"
    --rename_map="$RENAME_MAP_JSON"
    --policy.device=cuda
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --wandb.enable=false
  )
fi

# ---- print resolved configuration --------------------------------------------------------
echo "============ train_dessert_skill configuration ============="
echo "skill:            $SKILL"
echo "task:             $TASK"
echo "dataset root:     $DATASET_ROOT"
echo "dataset repo id:  $DATASET_REPO_ID"
echo "total episodes:   $TOTAL_EPISODES"
echo "total frames:     $TOTAL_FRAMES"
echo "pretrained model: $PRETRAINED_MODEL"
echo "  (local 2-camera derivative: $BASE_MODEL_DIR)"
echo "output dir:       $OUTPUT_DIR"
echo "job name:         $JOB_NAME"
echo "steps:            $STEPS"
echo "batch size:       $BATCH_SIZE"
echo "save frequency:   $SAVE_FREQ"
echo "resume:           $RESUME"
echo "device:           cuda"
echo "dry-run:          $DRY_RUN"
echo "============================================================="
echo
echo "Dataset camera keys:"
echo "- observation.images.front"
echo "- observation.images.side"
echo
echo "Policy camera keys:"
echo "- observation.images.camera1"
echo "- observation.images.camera2"
echo
echo "Applied rename map:"
echo "front -> camera1"
echo "side -> camera2"
echo
print_cmd "${CMD[@]}"

if [[ "$DRY_RUN" == true ]]; then
  echo "[dry-run] not executing."
  exit 0
fi

echo
echo "Training will occupy the GPU (hours at 20000 steps; ~minutes for a smoke test)."
read -r -p "Proceed with training? [y/N] " REPLY
if [[ "$REPLY" != "y" && "$REPLY" != "Y" ]]; then
  echo "Aborted."
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"
"${CMD[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/train_${SKILL}_$(date +%Y%m%d_%H%M%S).log"
