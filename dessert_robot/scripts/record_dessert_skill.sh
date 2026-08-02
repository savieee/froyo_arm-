#!/usr/bin/env bash
# Record teleoperated demonstrations for one dessert skill, ONE EPISODE PER
# lerobot-record INVOCATION. Episodes are NOT time-capped: each episode keeps
# recording until the operator types 'done'.
#
# How 'done' works (LeRobot 0.4.4's supported early-finish mechanism):
#   lerobot-record runs a global pynput keyboard listener; a RIGHT-ARROW press
#   sets events["exit_early"], which cleanly ends the recording loop, after
#   which lerobot saves the episode (dataset.save_episode()) and exits.
#   lerobot-record is launched as a managed background child so this wrapper
#   keeps the terminal; when the operator types 'done', the wrapper synthesizes
#   a Right-arrow key press with pynput.keyboard.Controller — the exact same
#   event a physical key tap produces. No signals are sent to end an episode.
#   (A physical Right-arrow tap in this desktop session still works too.)
#
# EMERGENCY FALLBACK ONLY: --dataset.episode_time_s is set to 3600 (1 hour).
# It exists solely so a forgotten/broken session cannot record forever; the
# normal way to end every episode is typing 'done'. An episode ended by this
# timeout is still saved by lerobot's normal path.
#
# Requires a graphical session (X11): LeRobot disables its keyboard listener
# entirely in headless environments, so 'done' would have nothing to trigger.
#
# Usage: ./scripts/record_dessert_skill.sh <skill> [num_episodes] [--resume] [--dry-run]
#   num_episodes omitted = keep recording until the operator quits with 'q'.
#
# Assumes the LeRobot conda/venv environment is already activated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=dessert_common.sh
source "${SCRIPT_DIR}/dessert_common.sh"

# Emergency safety cap per episode (seconds). NOT the normal ending mechanism.
EMERGENCY_EPISODE_TIME_S=3600

usage() {
  echo "Usage: $0 <skill> [num_episodes] [--resume] [--dry-run]" >&2
  echo "  skills: $(skills_list)" >&2
  echo "  num_episodes omitted: record until you answer q to 'Start the next episode?'" >&2
  exit 1
}

start_state_for_skill() {
  case "$1" in
    pick_spatula) cat <<'EOF'
  - robot starts empty-handed
  - spatula is at its standard location
  - main bowl and ingredient bowls are in their standard positions
EOF
      ;;
    scoop_strawberries) cat <<'EOF'
  - robot starts holding the spatula
  - destination bowl is empty
  - strawberries are in their standard bowl
EOF
      ;;
    scoop_blueberries) cat <<'EOF'
  - robot starts holding the spatula
  - strawberries are already in the destination bowl
  - blueberries are in their standard bowl
EOF
      ;;
    scoop_yogurt) cat <<'EOF'
  - robot starts holding the spatula
  - strawberries and blueberries are already in the destination bowl
  - yogurt is in its standard bowl
EOF
      ;;
    place_spatula) cat <<'EOF'
  - robot starts holding the spatula
  - destination bowl contains the previous ingredients
EOF
      ;;
    pour_honey) cat <<'EOF'
  - spatula has already been returned
  - robot starts empty-handed
  - honey container is at its standard location
EOF
      ;;
    place_honey) cat <<'EOF'
  - robot starts holding the honey container after pouring
EOF
      ;;
    return_home) cat <<'EOF'
  - honey container has already been returned
  - robot starts from the final pose of place_honey
EOF
      ;;
  esac
}

# ---- argument parsing ---------------------------------------------------------
SKILL=""
NUM_EPISODES=""
RESUME=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --resume)  RESUME=true ;;
    --dry-run) DRY_RUN=true ;;
    --*)       echo "ERROR: unsupported argument: $arg" >&2; usage ;;
    *)
      if [[ -z "$SKILL" ]]; then SKILL="$arg"
      elif [[ -z "$NUM_EPISODES" ]]; then NUM_EPISODES="$arg"
      else echo "ERROR: unexpected extra argument: $arg" >&2; usage
      fi
      ;;
  esac
done

[[ -n "$SKILL" ]] || usage

if ! TASK="$(task_for_skill "$SKILL")"; then
  echo "ERROR: unsupported skill '$SKILL' (expected: $(skills_list))" >&2
  exit 1
fi

if [[ -n "$NUM_EPISODES" ]] && ! is_positive_int "$NUM_EPISODES"; then
  echo "ERROR: num_episodes must be a positive integer, got '$NUM_EPISODES'" >&2
  exit 1
fi

require_pcan
require_cmd lerobot-record
require_cmd python

# The wrapper's 'done' command and lerobot's own episode-end listener both
# need pynput with a working display; without it there is no supported way to
# end an untimed episode. Skipped in dry-run (no processes are launched).
if [[ "$DRY_RUN" != true ]]; then
  if ! python -c "from pynput.keyboard import Controller, Key; Controller()" >/dev/null 2>&1; then
    echo "ERROR: pynput cannot control the keyboard in this session (headless or no X11?)." >&2
    echo "       LeRobot's episode-finish listener would also be disabled, so untimed" >&2
    echo "       recording is not possible here. Run from the desktop session." >&2
    exit 1
  fi
fi

# ---- dataset locations ----------------------------------------------------------
DATASET_ROOT="$PROJECT_ROOT/datasets/$SKILL"
DATASET_REPO_ID="seeed_rebot_b601_rs/dessert_$SKILL"

if [[ "$DRY_RUN" != true ]]; then
  mkdir -p "${PROJECT_ROOT}/datasets"
fi

# USE_RESUME tracks what the NEXT lerobot-record invocation needs:
# - empty dataset, no --resume: first episode creates the dataset (no resume
#   flag); every later episode this session resumes.
# - non-empty dataset: --resume is required up front; every invocation resumes.
# Never overwrite an existing dataset.
USE_RESUME=false
if [[ -d "$DATASET_ROOT" ]] && [[ -n "$(ls -A "$DATASET_ROOT" 2>/dev/null)" ]]; then
  if [[ "$RESUME" != true ]]; then
    echo "ERROR: dataset directory is not empty: $DATASET_ROOT" >&2
    echo "       Pass --resume to append episodes, or move/delete the directory." >&2
    exit 1
  fi
  USE_RESUME=true
elif [[ -d "$DATASET_ROOT" ]] && [[ "$DRY_RUN" != true ]]; then
  # LeRobot creates this directory itself (mkdir exist_ok=False); an empty
  # leftover directory would make it crash, so remove it.
  rmdir "$DATASET_ROOT"
fi

# ---- command construction --------------------------------------------------------
# One episode per invocation; reset time is minimal because this script (not
# lerobot-record) controls the pause between episodes. episode_time_s is the
# documented EMERGENCY fallback, not the normal episode-ending mechanism.
BASE_CMD=(
  lerobot-record
  --robot.type=seeed_b601_rs_follower
  --robot.port="$PCAN_IF"
  --robot.id=follower1
  --robot.can_adapter=socketcan
  --robot.cameras="$CAMERAS"
  --teleop.type=rebot_arm_102_leader
  --teleop.port=/dev/ttyUSB0
  --teleop.id=rebot_arm_102_leader
  --display_data=true
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.num_episodes=1
  --dataset.single_task="$TASK"
  --dataset.push_to_hub=false
  --dataset.episode_time_s="$EMERGENCY_EPISODE_TIME_S"
  --dataset.reset_time_s=1
)

# ---- managed-child helpers ---------------------------------------------------------
CHILD_PID=""
SESSION_TMP=""

# LeRobot 0.4.4 logs exactly "Recording episode <n>" (lerobot_record.py,
# log_say) the moment the robot, cameras and leader arm are connected and the
# recording loop begins. <n> is the current episode count, so on a resumed
# dataset it is nonzero — match the stable prefix, not "episode 0".
READINESS_PATTERN="Recording episode"
# Fallback ONLY if the readiness line never appears (e.g. changed log format):
# after this many seconds the prompt is shown anyway, with a warning.
READY_FALLBACK_S=120

cleanup_session() {
  # Never let the recorder outlive the wrapper. SIGINT gives lerobot-record its
  # normal KeyboardInterrupt shutdown; never SIGKILL.
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -INT "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  if [[ -n "$SESSION_TMP" && -d "$SESSION_TMP" ]]; then
    rm -rf "$SESSION_TMP"
  fi
}
trap cleanup_session EXIT
trap 'exit 130' INT

# Relays every child output line to the terminal unchanged (stdout) and
# creates the flag file the first time the readiness line is seen. Nothing is
# hidden or held back; this is a pass-through tap, not a filter.
readiness_monitor() {
  local flag="$1" line
  while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ ! -e "$flag" && "$line" == *"$READINESS_PATTERN"* ]]; then
      : > "$flag"
    fi
  done
}

send_finish_key() {
  # Synthesize the Right-arrow press that LeRobot 0.4.4's keyboard listener
  # maps to "exit early / finish this episode" (control_utils.py). This is the
  # supported episode-completion mechanism, identical to a physical key tap.
  python - <<'PYEOF'
from pynput.keyboard import Controller, Key
kb = Controller()
kb.press(Key.right)
kb.release(Key.right)
PYEOF
}

drain_stdin() {
  # The synthesized key press is also delivered to the focused window; if that
  # is this terminal, it arrives as an escape sequence on stdin. Discard any
  # such pending bytes so they cannot pollute the next prompt.
  local _junk
  while IFS= read -r -s -t 0.1 -n 1 _junk; do :; done
}

# Prints operator-facing text to stderr, on a fresh line, so it stays visually
# distinct from the child-process output relayed on stdout.
say_operator() {
  printf '%s\n' "$@" >&2
}

# Runs one lerobot-record invocation as a managed child and prompts the
# operator until they type 'done'. Returns non-zero if the recorder failed.
record_one_episode() {
  local ep="$1"; shift
  local ready_flag="${SESSION_TMP}/ready_ep${ep}"
  local finished_by_operator=false
  local status=0
  local reply
  local waited_ticks=0

  # All child output (stdout + stderr; the readiness line is a logging line on
  # stderr) streams live through readiness_monitor to the terminal.
  # PYTHONUNBUFFERED keeps the startup output real-time despite the pipe.
  PYTHONUNBUFFERED=1 "$@" > >(readiness_monitor "$ready_flag") 2>&1 &
  CHILD_PID=$!

  # --- phase 1: no prompts until LeRobot reports the recording loop started ---
  while kill -0 "$CHILD_PID" 2>/dev/null && [[ ! -e "$ready_flag" ]]; do
    sleep 0.2
    waited_ticks=$(( waited_ticks + 1 ))
    if (( waited_ticks >= READY_FALLBACK_S * 5 )); then
      say_operator "" \
        "WARNING: '${READINESS_PATTERN}' was not seen within ${READY_FALLBACK_S}s;" \
        "         showing the prompt anyway. Check the log output above."
      break
    fi
  done

  if kill -0 "$CHILD_PID" 2>/dev/null; then
    if [[ -e "$ready_flag" ]]; then
      say_operator "" \
        "==================================================" \
        "Robot and cameras are ready." \
        "Episode ${ep}${NUM_EPISODES:+ of ${NUM_EPISODES}} is now recording." \
        "=================================================="
    fi

    # --- phase 2: done/no loop ------------------------------------------------
    while true; do
      if ! kill -0 "$CHILD_PID" 2>/dev/null; then
        break  # recorder ended on its own (error, ESC key, or emergency timeout)
      fi
      printf '\nIs the episode done? Type done or no: ' >&2
      if ! read -r reply; then
        reply=""  # EOF on stdin: treat as invalid input and re-check the child
      fi
      if ! kill -0 "$CHILD_PID" 2>/dev/null; then
        break  # recorder ended while we were waiting for input
      fi
      case "$reply" in
        done)
          send_finish_key
          finished_by_operator=true
          say_operator "Finish signal sent; waiting for lerobot-record to save episode ${ep}..."
          break
          ;;
        no)
          ;;
        *)
          say_operator "Invalid input. Type done or no."
          ;;
      esac
    done
  fi

  wait "$CHILD_PID" || status=$?
  CHILD_PID=""
  drain_stdin

  if (( status != 0 )); then
    say_operator "ERROR: lerobot-record exited with status ${status}; episode ${ep} was NOT saved."
    return 1
  fi
  if [[ "$finished_by_operator" != true ]]; then
    say_operator \
      "WARNING: episode ${ep} ended without 'done' (ESC key, or the ${EMERGENCY_EPISODE_TIME_S}s emergency timeout)." \
      "         lerobot-record exited cleanly, so the episode was still saved."
  fi
  return 0
}

# ---- print resolved configuration --------------------------------------------------
echo "============ record_dessert_skill configuration ============"
echo "skill:            $SKILL"
echo "task:             $TASK"
echo "episodes:         ${NUM_EPISODES:-unlimited (stop with q)} (one lerobot-record run per episode)"
echo "episode end:      operator types 'done' (emergency cap: ${EMERGENCY_EPISODE_TIME_S}s)"
echo "dataset root:     $DATASET_ROOT"
echo "dataset repo id:  $DATASET_REPO_ID"
echo "PCAN interface:   $PCAN_IF"
echo "front camera:     $FRONT_CAMERA"
echo "side camera:      $SIDE_CAMERA"
echo "resume (session): $RESUME"
echo "dry-run:          $DRY_RUN"
echo "============================================================="
echo
echo "Physical starting state for every '$SKILL' episode:"
start_state_for_skill "$SKILL"

# ---- dry-run: print the managed-process flow, launch nothing -------------------------
if [[ "$DRY_RUN" == true ]]; then
  DEMO_EPISODES="${NUM_EPISODES:-2}"
  (( DEMO_EPISODES > 2 )) && DEMO_EPISODES=2
  for (( EP=1; EP<=DEMO_EPISODES; EP++ )); do
    EPISODE_CMD=( "${BASE_CMD[@]}" )
    [[ "$USE_RESUME" == true ]] && EPISODE_CMD+=(--resume=true)
    echo
    echo "[dry-run] episode ${EP}: resume=${USE_RESUME}"
    printf '[dry-run]   would launch as managed background child: '
    print_cmd "${EPISODE_CMD[@]}"
    echo "[dry-run]   would stream all LeRobot output live and wait for the readiness"
    echo "[dry-run]     line '${READINESS_PATTERN} <n>' before any operator prompt"
    echo "[dry-run]   then banner: Robot and cameras are ready. / Episode ${EP} is now recording."
    echo "[dry-run]   would prompt (stderr): Is the episode done? Type done or no:"
    echo "[dry-run]   'no' / invalid input -> keep recording, ask again"
    echo "[dry-run]   'done' -> synthesize Right-arrow (pynput) -> lerobot exit_early"
    echo "[dry-run]             -> episode saved via dataset.save_episode() -> child exits"
    echo "[dry-run]   then: Start the next episode? [y/q] (y -> 10s countdown, q -> stop)"
    USE_RESUME=true
  done
  if [[ -z "$NUM_EPISODES" ]] || (( NUM_EPISODES > DEMO_EPISODES )); then
    echo
    echo "[dry-run] ... pattern repeats for every further episode (all with --resume=true)."
  fi
  echo "[dry-run] no processes launched, no hardware touched, nothing written."
  exit 0
fi

# ---- episode loop -------------------------------------------------------------------
SESSION_TMP="$(mktemp -d -t record_dessert_skill.XXXXXX)"  # readiness flags; removed on exit
RECORDED=0
EP=1
while true; do
  echo
  echo "Preparing episode ${EP}${NUM_EPISODES:+ of ${NUM_EPISODES}} for ${SKILL}"

  EPISODE_CMD=( "${BASE_CMD[@]}" )
  if [[ "$USE_RESUME" == true ]]; then
    EPISODE_CMD+=(--resume=true)
  fi

  for (( SEC=10; SEC>=1; SEC-- )); do
    echo "Starting episode ${EP} in ${SEC}..."
    sleep 1
  done
  echo "Starting episode ${EP} now. Waiting for the robot, cameras and leader arm to connect..."

  if ! record_one_episode "$EP" "${EPISODE_CMD[@]}"; then
    echo "Stopping the session because the recorder failed." >&2
    echo "Episodes recorded this session: ${RECORDED}."
    echo "Dataset root: $DATASET_ROOT"
    exit 1
  fi

  RECORDED=$(( RECORDED + 1 ))
  USE_RESUME=true
  echo "Episode ${EP}${NUM_EPISODES:+ of ${NUM_EPISODES}} saved successfully."

  if [[ -n "$NUM_EPISODES" ]] && (( RECORDED >= NUM_EPISODES )); then
    break
  fi

  while true; do
    read -r -p "Start the next episode? [y/q] " REPLY || REPLY=""
    case "$REPLY" in
      y|Y) break ;;
      q|Q)
        echo
        echo "Recording session stopped by operator."
        echo "Episodes recorded this session: ${RECORDED}."
        echo "Dataset root: $DATASET_ROOT"
        echo "Dataset repo ID: $DATASET_REPO_ID"
        exit 0
        ;;
      *)   echo "Please answer y (record next episode) or q (quit)." ;;
    esac
  done
  EP=$(( EP + 1 ))
done

echo
echo "Recording session complete."
echo "Episodes recorded this session: ${RECORDED}"
echo "Dataset root: $DATASET_ROOT"
echo "Dataset repo ID: $DATASET_REPO_ID"
