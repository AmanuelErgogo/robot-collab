#!/usr/bin/env python
"""Compare two RoCoBench runs on paired variations."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.rocobench_pack_v1.metrics import paired_comparison
from benchmark.rocobench_pack_v1.results import read_episodes_csv


def _episodes_path(value: str) -> str:
    if os.path.isdir(value):
        return os.path.join(value, "episodes.csv")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    comparison = paired_comparison(read_episodes_csv(_episodes_path(args.run_a)), read_episodes_csv(_episodes_path(args.run_b)))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, sort_keys=True)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

