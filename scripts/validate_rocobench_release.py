#!/usr/bin/env python
"""Validate the Phase 8 benchmark release."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.rocobench_pack_v1.validator import validate_release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-bridge-smoke", action="store_true", help="Run live reset/step/render against the Phase 0 bridge.")
    parser.add_argument("--require-bridge", action="store_true", help="Fail if the live bridge smoke cannot run.")
    parser.add_argument("--output", default="artifacts/benchmark/release_validation.json")
    args = parser.parse_args(argv)
    result = validate_release(
        run_bridge_smoke=args.run_bridge_smoke,
        require_bridge=args.require_bridge,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

