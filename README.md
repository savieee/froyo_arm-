# FROYO Arm 

Yogurt dessert (Blueberry + Strawberry 😁) assembly robot built at the Revolute Hackathon (Aug 2026) with a
Seeed reBot B601-RS follower arm, a reBot 102 leader arm, and SmolVLA fine-tuned
via [LeRobot](https://github.com/Seeed-Projects/lerobot).

The robot prepares a frozen yogurt dessert using four learned manipulation skills:

1. `pick_spatula`
2. `scoop_strawberries`
3. `scoop_blueberries`
4. `scoop_yogurt`

## Repository layout

| Path | Contents |
|---|---|
| `dessert_robot/scripts/` | Recording, training, inference, and orchestration scripts |
| `dessert_robot/datasets/` | Local LeRobot datasets organized by skill |
| `dessert_robot/checkpoints/` | Local SmolVLA checkpoints organized by skill |
| `lerobot/` | Vendored Seeed-Projects LeRobot 0.4.4 fork |
| `outputs/captured_images/` | Temporary camera-discovery and capture outputs |

Large datasets and model checkpoints are not committed to GitHub. The trained
SmolVLA checkpoint includes an approximately large `model.safetensors` file,
which exceeds GitHub's regular 100 MB per file limit.

## Download datasets and trained models

The recorded LeRobot datasets and trained SmolVLA checkpoints are stored
externally on Google Drive.

- [Download the LeRobot datasets](https://drive.google.com/drive/folders/15Xp5a2qQ7_mred4Armc371waCnF-oNss?usp=drive_link)
- [Download the trained SmolVLA checkpoints](https://drive.google.com/drive/folders/1_12wk3vIbbuFmfgGX6GtQ6jQ8-YoaZC2?usp=drive_link)

After downloading, extract the files so that the project has the following
structure:

```text
dessert_robot/
├── datasets/
│   ├── pick_spatula/
│   ├── scoop_strawberries/
│   ├── scoop_blueberries/
│   └── scoop_yogurt/
│
└── checkpoints/
    ├── pick_spatula_checkpoints/
    ├── scoop_strawberries_checkpoints/
    ├── scoop_blueberries_checkpoints/
    └── scoop_yogurt_checkpoints/
```

## Setup

Follow the official Seeed Studio setup guide for installing LeRobot, configuring
the reBot B601-RS follower arm, calibrating the leader and follower arms, and
setting up the cameras:

[Getting Started with LeRobot-based reBot Arm B601-RS](https://wiki.seeedstudio.com/rebot_arm_b601_rs_lerobot/)

This project uses the Seeed LeRobot fork included under `lerobot/`. After
completing the system and hardware prerequisites in the guide, clone this
repository and install the local LeRobot package:

```bash
git clone https://github.com/savieee/froyo_arm-.git
cd froyo_arm-

conda activate lerobot
pip install -e ./lerobot
```

## Environment Setup

<img width="2000" height="924" alt="image" src="https://github.com/user-attachments/assets/5a0a5041-4325-43ea-b2b4-518802964fda" />


## Workflow

<img width="1795" height="617" alt="image" src="https://github.com/user-attachments/assets/4ab1ee6b-5988-4d6c-920c-43902bf8a78d" />


Move into the dessert robot directory on the lerobot conda env:

### 1. Record demonstrations for a skill (tele-operated)

Generic command:

```bash
./scripts/record_dessert_skill.sh <skill_name>
```

Example:

```bash
./scripts/record_dessert_skill.sh pick_spatula
```

During recording, the reBot 102 leader arm is used to teleoperate the B601-RSfollower arm. Each episode stores:

- Front camera observations
- Side camera observations
- Follower-arm state
- Leader-arm actions
- Timestamps
- Task instruction
- Episode and frame indices

### 2. Fine tune SmolVLA for the recorded skills

Generic command:

```bash
./scripts/train_dessert_skill.sh <skill_name> \
  --steps <training_steps> \
  --batch-size <batch_size> \
  --save-freq <checkpoint_frequency>
```

Example:

```bash
./scripts/train_dessert_skill.sh scoop_blueberries \
  --steps 5000 \
  --batch-size 2 \
  --save-freq 5000
```

Training outputs are written to the corresponding skill directory under:

```text
dessert_robot/checkpoints/
```
### 3. Run and test one trained skill

Generic command:

```bash
./scripts/run_dessert_skill.sh <skill_name>
```

Example:

```bash
./scripts/run_dessert_skill.sh scoop_blueberries
```

### 4. Run the full sequence

The orchestrator executes the four separately trained SmolVLA policies in
sequence:

```bash
cd ~/froyo_arm-/dessert_robot
conda activate lerobot
export PCAN_IF=can0
export PYTHONNOUSERSITE=1

./scripts/run_dessert_orchestrator.sh \
  --pick-spatula 1 \
  --scoop-strawberries 1 \
  --scoop-blueberries 1 \
  --scoop-yogurt 1 \
  --pick-spatula-duration 15 \
  --scoop-strawberries-duration 30 \
  --scoop-blueberries-duration 30 \
  --scoop-yogurt-duration 29 \
  --yes
```

A value of `1` enables a skill, while `0` skips it:

```text
1 → Run the skill
0 → Skip the skill
```

## Hardware notes

- Follower: Seeed reBot B601-RS
- Follower communication: SocketCAN at 1 Mbit/s
- Leader: Seeed reBot 102 used for teleoperation and data collection
- Cameras: front and side USB cameras
- Policy: SmolVLA fine-tuned using LeRobot
- Training GPU: single GPU with approximately 7.6 GB of available memory
- Training batch size: 1–2, depending on available GPU memory

## Contributors:

- Savita Kendre
- Sairam Sridharan
- Ahilesh Vadivel
- Jaime Romero
- Sachidanand Halhalli
