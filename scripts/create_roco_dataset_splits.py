#!/usr/bin/env python
"""Create immutable variation-group dataset splits."""

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.dataset.cli import build_split_parser, create_splits_cli


def main() -> int:
    return create_splits_cli(build_split_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
