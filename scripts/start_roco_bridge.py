#!/usr/bin/env python
"""Start the RoCo Phase 0 bridge server."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.roco_runtime.cli import main


if __name__ == "__main__":
    main()
