# dessert_robot

Yogurt-dessert assembly with a Seeed reBot B601-RS follower arm, a reBot 102
leader arm, and LeRobot 0.4.4 (SmolVLA). The dessert is built from **eight
skills**, always in this order:

| # | Skill | Language instruction |
|---|-------|----------------------|
| 1 | `pick_spatula` | Pick up the spatula. |
| 2 | `scoop_strawberries` | Scoop strawberries into the bowl. |
| 3 | `scoop_blueberries` | Scoop blueberries into the bowl. |
| 4 | `scoop_yogurt` | Scoop yogurt into the bowl. |
| 5 | `place_spatula` | Place the spatula back. |
| 6 | `pour_honey` | Pour honey into the bowl. |
| 7 | `place_honey` | Place the honey back. |
| 8 | `return_home` | Return to the home position. |

## Layout

```
dessert_robot/
├── README.md
├── scripts/
│   ├── dessert_common.sh            # shared skills/tasks/cameras/validation helpers
│   ├── dataset_info.py              # opens a dataset with the LeRobot API, prints JSON summary
│   ├── merge_dessert_datasets.py    # official-API dataset merging (merge_datasets)
│   ├── record_dessert_skill.sh      # teleop recording, one episode per invocation
│   ├── build_full_dessert_dataset.sh# merge all 8 skill datasets into one multitask dataset
│   ├── train_dessert_skill.sh       # fine-tune SmolVLA on one skill
│   ├── train_dessert_vla.sh         # fine-tune ONE multitask SmolVLA on the combined dataset
│   ├── run_dessert_skill.sh         # run one skill autonomously (single or full checkpoint)
│   └── run_full_dessert.sh          # run all 8 skills in order, operator-confirmed
├── datasets/      # datasets/<skill>/ and datasets/final_full_dessert/
├── checkpoints/   # checkpoints/<skill>/ and checkpoints/final_full_dessert/
├── evaluations/   # evaluations/<single|full>/<skill>/<timestamp>/ per autonomous run
└── logs/          # training logs
```

## Prerequisites (every terminal)

```bash
conda activate lerobot
export PCAN_IF=can0
```

If the CAN adapter was unplugged/bumped, bring the interface back up first:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
```

Cameras default to the stable `/dev/v4l/by-id/...` paths for the front
(SN0002) and side (G720P) cameras; override with `FRONT_CAMERA` / `SIDE_CAMERA`
environment variables if the hardware changes. Never use camera index 0 — it
is a v4l2loopback device that hangs OpenCV.

Every script supports `--dry-run` (prints the exact commands, touches no
hardware and writes no data).

## Workflow A — one skill end to end

Use this to validate the full pipeline on a single skill before investing in
all eight.

```bash
# 1. Record 20 demonstrations (one lerobot-record run per episode; y/q prompt
#    and a 10-second countdown before each). Re-run later with --resume to
#    append more episodes.
./scripts/record_dessert_skill.sh scoop_strawberries 20

# 2. Smoke-test training (~minutes).
./scripts/train_dessert_skill.sh scoop_strawberries --steps 100 --batch-size 1 --save-freq 100
```

The smoke test validates dataset loading, model loading, forward pass,
backward pass, GPU compatibility, and checkpoint saving. It is not expected to
produce a useful policy.

```bash
# 3. Real training (delete or move checkpoints/scoop_strawberries/ first if the
#    smoke test wrote there, or continue it with --resume).
./scripts/train_dessert_skill.sh scoop_strawberries --steps 20000

# 4. Run the skill autonomously with its single-skill checkpoint.
./scripts/run_dessert_skill.sh scoop_strawberries --checkpoint-mode single
```

## Workflow B — full dessert with one multitask VLA

```bash
# 1. Record all eight skills (set up the physical starting state the script
#    prints for each skill).
./scripts/record_dessert_skill.sh pick_spatula 20
./scripts/record_dessert_skill.sh scoop_strawberries 20
./scripts/record_dessert_skill.sh scoop_blueberries 20
./scripts/record_dessert_skill.sh scoop_yogurt 20
./scripts/record_dessert_skill.sh place_spatula 20
./scripts/record_dessert_skill.sh pour_honey 20
./scripts/record_dessert_skill.sh place_honey 20
./scripts/record_dessert_skill.sh return_home 20

# 2. Build the combined multitask dataset (official LeRobot merge API only).
./scripts/build_full_dessert_dataset.sh

# 3. Smoke-test the multitask training (same statement as above applies).
./scripts/train_dessert_vla.sh --steps 100 --batch-size 1 --save-freq 100

# 4. Full multitask training (clear/resume checkpoints/final_full_dessert first
#    if the smoke test used it).
./scripts/train_dessert_vla.sh --steps 20000

# 5. Sanity-check ONE instruction with the multitask checkpoint.
./scripts/run_dessert_skill.sh scoop_strawberries --checkpoint-mode full

# 6. Run the whole dessert.
./scripts/run_full_dessert.sh
```

## Rules and caveats

- **Rebuild after new recordings.** The combined dataset is a snapshot. If you
  add episodes to any skill, re-run `./scripts/build_full_dessert_dataset.sh
  --force` and retrain the VLA; the old combined dataset does not update
  itself.
- **One checkpoint, all instructions.** `train_dessert_vla.sh` produces a
  single multitask checkpoint that serves every language instruction; the
  instruction passed at run time selects the behavior.
- **The orchestrator only sequences.** `run_full_dessert.sh` runs the skills
  in order and asks after each one: `y` = next skill, `r` = retry the same
  skill, `q` = stop. It never resets the scene, retries automatically, or runs
  anything in the background — restoring the scene between attempts is on you.
- **Confirmation means physical success.** Answer `y` only if the skill
  actually succeeded on the table. The policy has no way to know whether the
  scoop really landed in the bowl.
- **Datasets are never overwritten silently.** Recording into a non-empty
  dataset requires `--resume`; rebuilding the combined dataset requires
  `--force` plus typing `DELETE`; training over existing checkpoints requires
  `--resume` or manual cleanup.
- **Autonomous runs are gated.** `run_dessert_skill.sh` requires typing `RUN`,
  `run_full_dessert.sh` requires typing `START DESSERT`. Keep the emergency
  stop in reach; Ctrl+C is a normal way to stop.
- Each autonomous run records a single evaluation episode into a fresh
  timestamped folder under `evaluations/<mode>/<skill>/` for later review with
  `lerobot-dataset-viz` (pass `--display-compressed-images false`).
