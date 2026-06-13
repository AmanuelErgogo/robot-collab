#!/usr/bin/env python
"""Inspect a Phase 3 ACT checkpoint without launching simulator rollout."""

import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.dataset.manifest import atomic_write_json
from integrations.lerobot_roco.training.checkpoint import inspect_checkpoint, run_clean_metadata_reload


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect RoCo ACT checkpoint metadata.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metadata-only", action="store_true", help="Do not spawn a clean inspection subprocess.")
    args = parser.parse_args()

    inspection = inspect_checkpoint(args.checkpoint_dir, dataset_root=args.dataset_root)
    payload = inspection.to_dict()
    if not args.metadata_only:
        payload["clean_metadata_reload_exit_code"] = run_clean_metadata_reload(args.checkpoint_dir, args.dataset_root)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        atomic_write_json(os.path.join(args.output_dir, "checkpoint_inspection.json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if inspection.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
