#!/usr/bin/env python
"""Compute Phase 3 offline action diagnostics from saved predictions."""

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.training.evaluate_offline import main


if __name__ == "__main__":
    raise SystemExit(main())
