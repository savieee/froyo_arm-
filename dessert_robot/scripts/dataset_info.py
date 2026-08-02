#!/usr/bin/env python
"""Print a JSON summary of a local LeRobot dataset (official metadata API only).

Usage:
    python dataset_info.py --root PATH --repo-id ID [--require-features] [--require-tasks T1,T2,...]

Exit codes: 0 ok, 1 cannot open / validation failed.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--require-features",
        action="store_true",
        help="fail unless front/side cameras, observation.state and action are present",
    )
    parser.add_argument(
        "--require-tasks",
        default=None,
        help="comma-separated task strings that must each have at least one episode",
    )
    args = parser.parse_args()

    # Fail fast on missing local data; without this, LeRobot falls back to
    # querying the HuggingFace Hub for the repo_id over the network.
    if not (Path(args.root) / "meta").is_dir():
        print(f"ERROR: not a local LeRobot dataset (no meta/ directory): {args.root}", file=sys.stderr)
        return 1

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    except Exception as exc:  # noqa: BLE001 - report any open failure clearly
        print(f"ERROR: cannot open dataset at {args.root}: {exc}", file=sys.stderr)
        return 1

    features = meta.features
    camera_keys = list(meta.camera_keys)

    episodes_per_task: dict[str, int] = {}
    try:
        for ep_tasks in meta.episodes["tasks"]:
            for task in ep_tasks:
                episodes_per_task[task] = episodes_per_task.get(task, 0) + 1
    except Exception:
        # Fall back to the task table without per-episode counts.
        episodes_per_task = dict.fromkeys(list(meta.tasks.index), -1)

    info = {
        "root": str(meta.root),
        "repo_id": args.repo_id,
        "total_episodes": meta.total_episodes,
        "total_frames": meta.total_frames,
        "fps": meta.fps,
        "features": {name: list(spec.get("shape", [])) for name, spec in features.items()},
        "camera_keys": camera_keys,
        "state_shape": list(features.get("observation.state", {}).get("shape", [])),
        "action_shape": list(features.get("action", {}).get("shape", [])),
        "tasks": episodes_per_task,
    }
    print(json.dumps(info, indent=2))

    ok = True
    if meta.total_episodes < 1:
        print("ERROR: dataset has no episodes", file=sys.stderr)
        ok = False

    if args.require_features:
        for key in ("observation.images.front", "observation.images.side"):
            if key not in camera_keys:
                print(f"ERROR: missing camera feature: {key}", file=sys.stderr)
                ok = False
        for key in ("observation.state", "action"):
            if key not in features:
                print(f"ERROR: missing feature: {key}", file=sys.stderr)
                ok = False
        if not episodes_per_task:
            print("ERROR: no task labels present", file=sys.stderr)
            ok = False

    if args.require_tasks:
        required = [t for t in args.require_tasks.split(",") if t]
        for task in required:
            if episodes_per_task.get(task, 0) == 0:
                print(f"ERROR: task has zero episodes or is absent: {task!r}", file=sys.stderr)
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
