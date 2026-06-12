#!/usr/bin/env python
"""Report LeRobot preprocessing compatibility for the RoCo Gym environment."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
CLIENT_SRC = os.path.join(REPO_ROOT, "integrations", "lerobot_roco", "client", "src")
if CLIENT_SRC not in sys.path:
    sys.path.insert(0, CLIENT_SRC)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test LeRobot observation preprocessing compatibility.")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--active-agent", default="Alice")
    args = parser.parse_args()
    try:
        from lerobot_roco_env import RoCoGymEnv
        from lerobot_roco_env.lerobot_compat import get_lerobot_compatibility_report

        env = RoCoGymEnv(endpoint=args.endpoint, active_agent=args.active_agent)
        report = get_lerobot_compatibility_report(env)
        print(json.dumps(report, indent=2, sort_keys=True))
        env.close()
        return 0 if report.get("lerobot_version") else 2
    except Exception as exc:
        print("LeRobot preprocessing smoke test failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
