#!/usr/bin/env python
"""Collect RRT expert demonstrations for the Phase 2 dataset."""

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.dataset.cli import build_collect_parser, collect_dataset


def main() -> int:
    return collect_dataset(build_collect_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
