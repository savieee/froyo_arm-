#!/usr/bin/env python
"""Merge the eight dessert skill datasets into one multitask dataset.

Uses ONLY official LeRobot 0.4.4 APIs:
  - lerobot.datasets.lerobot_dataset.LeRobotDataset / LeRobotDatasetMetadata
  - lerobot.datasets.dataset_tools.merge_datasets

Usage:
    python merge_dessert_datasets.py \
        --sources root1:repo_id1 root2:repo_id2 ... \
        --output-root PATH --output-repo-id ID [--dry-run]

Source order on the command line is preserved in the merged dataset.
Validation: every source must open, and FPS, feature schema, camera keys,
state/action dims must match across sources. Sources are never modified.

Exit codes: 0 ok, 1 validation or merge failure.
"""

import argparse
import json
import sys
from pathlib import Path


def summarize(meta):
    features = meta.features
    return {
        "fps": meta.fps,
        "features": {name: list(spec.get("shape", [])) for name, spec in features.items()},
        "camera_keys": sorted(meta.camera_keys),
        "state_shape": list(features.get("observation.state", {}).get("shape", [])),
        "action_shape": list(features.get("action", {}).get("shape", [])),
    }


def episodes_per_task(meta):
    counts: dict[str, int] = {}
    for ep_tasks in meta.episodes["tasks"]:
        for task in ep_tasks:
            counts[task] = counts.get(task, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, metavar="ROOT:REPO_ID")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    output_root = Path(args.output_root)
    if output_root.exists():
        print(f"ERROR: output already exists: {output_root}", file=sys.stderr)
        return 1

    # ---- validate all sources -------------------------------------------------
    sources = []
    for spec in args.sources:
        root, sep, repo_id = spec.partition(":")
        if not sep or not root or not repo_id:
            print(f"ERROR: bad --sources entry (want ROOT:REPO_ID): {spec}", file=sys.stderr)
            return 1
        sources.append((Path(root), repo_id))

    reference = None
    total_src_episodes = 0
    for root, repo_id in sources:
        # Fail fast on missing local data; without this, LeRobot falls back to
        # querying the HuggingFace Hub for the repo_id over the network.
        if not (root / "meta").is_dir():
            print(f"ERROR: not a local LeRobot dataset (no meta/ directory): {root}", file=sys.stderr)
            return 1
        try:
            meta = LeRobotDatasetMetadata(repo_id, root=root)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: cannot open source dataset {repo_id} at {root}: {exc}", file=sys.stderr)
            return 1
        if meta.total_episodes < 1:
            print(f"ERROR: source dataset has no episodes: {root}", file=sys.stderr)
            return 1
        summary = summarize(meta)
        total_src_episodes += meta.total_episodes
        print(f"source ok: {repo_id}  episodes={meta.total_episodes} frames={meta.total_frames} fps={meta.fps}")
        if reference is None:
            reference = (repo_id, summary)
        else:
            ref_id, ref = reference
            for key in ("fps", "features", "camera_keys", "state_shape", "action_shape"):
                if summary[key] != ref[key]:
                    print(
                        f"ERROR: {key} mismatch between {ref_id} and {repo_id}:\n"
                        f"  {ref_id}: {ref[key]}\n  {repo_id}: {summary[key]}",
                        file=sys.stderr,
                    )
                    return 1

    if args.dry_run:
        print(f"[dry-run] would merge {len(sources)} datasets ({total_src_episodes} episodes) "
              f"into {output_root} as {args.output_repo_id}")
        return 0

    # ---- merge (official API; preserves per-episode task labels, reindexes) ----
    datasets = [LeRobotDataset(repo_id, root=root) for root, repo_id in sources]
    merge_datasets(datasets, output_repo_id=args.output_repo_id, output_dir=output_root)

    # ---- reopen combined dataset and report -------------------------------------
    merged = LeRobotDatasetMetadata(args.output_repo_id, root=output_root)
    counts = episodes_per_task(merged)
    report = {
        "output_root": str(output_root),
        "output_repo_id": args.output_repo_id,
        "total_episodes": merged.total_episodes,
        "total_frames": merged.total_frames,
        "fps": merged.fps,
        "features": {name: list(spec.get("shape", [])) for name, spec in merged.features.items()},
        "camera_keys": list(merged.camera_keys),
        "state_shape": list(merged.features.get("observation.state", {}).get("shape", [])),
        "action_shape": list(merged.features.get("action", {}).get("shape", [])),
        "episodes_per_task": counts,
    }
    print(json.dumps(report, indent=2))

    if merged.total_episodes != total_src_episodes:
        print(
            f"ERROR: merged episode count {merged.total_episodes} != sum of sources {total_src_episodes}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
