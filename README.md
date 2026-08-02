# froyo_arm

Yogurt-dessert assembly robot built at the Revolute Hackathon (Aug 2026) with a
Seeed reBot B601-RS follower arm, a reBot 102 leader arm, and SmolVLA fine-tuned
via [LeRobot](https://github.com/Seeed-Projects/lerobot).

The robot assembles a frozen-yogurt dessert as a sequence of 8 learned skills:

1. `pick_spatula`
2. `scoop_strawberries`
3. `scoop_blueberries`
4. `scoop_yogurt`
5. `place_spatula`
6. `pour_honey`
7. `place_honey`
8. `return_home`

## Repository layout

| Path | Contents |
|------|----------|
| `dessert_robot/scripts/` | Record / train / run shell scripts (all support `--dry-run`) |
| `dessert_robot/datasets/` | Recorded teleoperation episodes per skill (LeRobot dataset format, videos included) |
| `lerobot/` | Vendored Seeed-Projects LeRobot 0.4.4 fork, including a local `camera_opencv.py` fix for a `/dev/video0` loopback device that hangs OpenCV |
| `outputs/captured_images/` | Scratch output directory for camera captures |

Training checkpoints, generated models, and evaluation runs are intentionally
not committed (`dessert_robot/checkpoints/`, `models/`, `evaluations/`): the
fine-tuned SmolVLA checkpoint is an 865 MB safetensors file, over GitHub's
100 MB limit.

## Setup

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ./lerobot
export PCAN_IF=can0          # CAN interface for the follower arm
export PYTHONNOUSERSITE=1    # avoid user-site package conflicts
```

Cameras are addressed by stable `/dev/v4l/by-id/` paths (front + side USB
cameras); see `dessert_robot/scripts/dessert_common.sh`.

## Workflow

```bash
cd dessert_robot

# 1. Record episodes for a skill (operator-controlled, --resume to append)
./scripts/record_dessert_skill.sh pick_spatula

# 2. Train a single skill (SmolVLA fine-tune)
./scripts/train_dessert_skill.sh pick_spatula

# 3. Run a skill autonomously from its checkpoint
./scripts/run_dessert_skill.sh pick_spatula

# Or: merge all 8 skill datasets and train one multitask policy
./scripts/build_full_dessert_dataset.sh
./scripts/train_dessert_vla.sh
./scripts/run_full_dessert.sh
```

The datasets use camera keys `front`/`side`, while the `lerobot/smolvla_base`
checkpoint expects `camera1`/`camera2`/`camera3`. The training scripts handle
this automatically: they generate a local 2-camera derivative of the base model
and apply `--rename_map` at train time; the run scripts apply the matching
mapping at inference. Dataset files are never modified.

## Hardware notes

- Follower: Seeed reBot B601-RS on CAN (`socketcan`, 1 Mbit/s)
- Leader: Seeed reBot 102 (teleoperation during recording)
- GPU: single 7.6 GB card — batch size 1 for SmolVLA fine-tuning
