#!/usr/bin/env python
"""Launch the Phase 3 tiny-set ACT overfit config."""

import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.training.config import overfit_training_config
from integrations.lerobot_roco.training.launch import TrainingLaunchError, launch_training
from integrations.lerobot_roco.training.report import write_phase3_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 3 controlled ACT overfit launch.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--lerobot-dataset-root", default=None)
    args = parser.parse_args()

    config = overfit_training_config()
    overrides = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.overwrite:
        overrides["overwrite"] = True
    if args.dataset_root:
        overrides["dataset_root"] = args.dataset_root
    if args.lerobot_dataset_root:
        overrides["lerobot_dataset_root"] = args.lerobot_dataset_root
    if overrides:
        config = config.with_overrides(**overrides)

    try:
        result = launch_training(config)
        write_phase3_report(result.run_dir)
    except TrainingLaunchError as exc:
        run_dir = os.path.abspath(config.output_dir)
        if os.path.exists(os.path.join(run_dir, "run_manifest.json")):
            write_phase3_report(run_dir)
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
