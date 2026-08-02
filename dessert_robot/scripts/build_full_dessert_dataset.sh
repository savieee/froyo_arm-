#!/usr/bin/env bash
# Build the combined multitask dataset from all eight dessert skill datasets,
# using ONLY the official LeRobot dataset merge API (via merge_dessert_datasets.py).
#
# Usage: ./scripts/build_full_dessert_dataset.sh [--force] [--dry-run]
#
# Assumes the LeRobot conda/venv environment is already activated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=dessert_common.sh
source "${SCRIPT_DIR}/dessert_common.sh"

usage() {
  echo "Usage: $0 [--force] [--dry-run]" >&2
  exit 1
}

FORCE=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --force)   FORCE=true ;;
    --dry-run) DRY_RUN=true ;;
    *)         echo "ERROR: unsupported argument: $arg" >&2; usage ;;
  esac
done

require_cmd python

OUTPUT_ROOT="$PROJECT_ROOT/datasets/final_full_dessert"
OUTPUT_REPO_ID="seeed_rebot_b601_rs/final_full_dessert"

# ---- validate sources exist (merge order = dessert sequence order) --------------
SOURCES=()
MISSING=false
for SKILL in "${DESSERT_SKILLS[@]}"; do
  ROOT="$PROJECT_ROOT/datasets/$SKILL"
  if [[ ! -d "$ROOT" ]] || [[ -z "$(ls -A "$ROOT" 2>/dev/null)" ]]; then
    echo "ERROR: source dataset missing or empty: $ROOT" >&2
    echo "       Record it first: ./scripts/record_dessert_skill.sh $SKILL <num_episodes>" >&2
    MISSING=true
  fi
  SOURCES+=("${ROOT}:seeed_rebot_b601_rs/dessert_${SKILL}")
done
if [[ "$MISSING" == true ]]; then
  exit 1
fi

# ---- overwrite protection ---------------------------------------------------------
if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ "$FORCE" != true ]]; then
    echo "ERROR: output already exists: $OUTPUT_ROOT" >&2
    echo "       Re-run with --force to delete and rebuild it." >&2
    exit 1
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] would delete existing output: $OUTPUT_ROOT"
  else
    echo "About to DELETE the existing combined dataset: $OUTPUT_ROOT"
    read -r -p "Type DELETE to confirm: " REPLY
    if [[ "$REPLY" != "DELETE" ]]; then
      echo "Aborted."
      exit 1
    fi
    rm -rf "$OUTPUT_ROOT"
  fi
fi

# ---- build merge command ------------------------------------------------------------
MERGE_CMD=(
  python "${SCRIPT_DIR}/merge_dessert_datasets.py"
  --sources "${SOURCES[@]}"
  --output-root "$OUTPUT_ROOT"
  --output-repo-id "$OUTPUT_REPO_ID"
)
if [[ "$DRY_RUN" == true ]]; then
  MERGE_CMD+=(--dry-run)
fi

# ---- print resolved configuration -----------------------------------------------------
echo "========= build_full_dessert_dataset configuration ========="
echo "merge order:"
for SKILL in "${DESSERT_SKILLS[@]}"; do
  echo "  - $SKILL"
done
echo "output root:    $OUTPUT_ROOT"
echo "output repo id: $OUTPUT_REPO_ID"
echo "force:          $FORCE"
echo "dry-run:        $DRY_RUN"
echo "============================================================="
print_cmd "${MERGE_CMD[@]}"

# The Python helper validates every source (openable, matching FPS/schema/
# cameras/state/action), merges with the official API, then reopens the result
# and prints episodes, frames, fps, features, cameras, shapes and per-task
# episode counts. In --dry-run it validates and reports without writing.
"${MERGE_CMD[@]}"

if [[ "$DRY_RUN" == true ]]; then
  echo "[dry-run] no files were created, removed, or modified."
  exit 0
fi

# ---- final validation: every dessert task must be present with >=1 episode ---------
REQUIRED_TASKS=""
for SKILL in "${DESSERT_SKILLS[@]}"; do
  TASK="$(task_for_skill "$SKILL")"
  REQUIRED_TASKS+="${REQUIRED_TASKS:+,}${TASK}"
done

echo
echo "Verifying combined dataset..."
if ! dataset_info "$OUTPUT_ROOT" "$OUTPUT_REPO_ID" --require-features --require-tasks "$REQUIRED_TASKS" >/dev/null; then
  echo "ERROR: combined dataset failed validation (see errors above)." >&2
  exit 1
fi
echo "Combined dataset OK: $OUTPUT_ROOT ($OUTPUT_REPO_ID)"
