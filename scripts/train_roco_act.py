#!/usr/bin/env python
"""Launch Phase 3 ACT training through the LeRobot CLI."""

import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.training.config import load_training_config
from integrations.lerobot_roco.training.launch import TrainingLaunchError, launch_training
from integrations.lerobot_roco.training.report import write_phase3_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch RoCo Phase 3 ACT training.")
    parser.add_argument("--config", required=True, help="Phase 3 training YAML/JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Write preflight/manifests but do not run training.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing run manifest.")
    parser.add_argument("--no-require-lerobot-dataset", action="store_true")
    parser.add_argument("--no-require-lerobot-package", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config)
    overrides = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.overwrite:
        overrides["overwrite"] = True
    if args.no_require_lerobot_dataset:
        overrides["require_lerobot_dataset"] = False
    if args.no_require_lerobot_package:
        overrides["require_lerobot_package"] = False
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
