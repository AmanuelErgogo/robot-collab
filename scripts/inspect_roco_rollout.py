#!/usr/bin/env python
"""Inspect Phase 4 rollout artifacts for alignment and queue traces."""

import argparse
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_jsonl(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def inspect_rollout(artifact_dir):
    result = _read_json(os.path.join(artifact_dir, "result.json"))
    with np.load(os.path.join(artifact_dir, "actions.npz")) as actions_npz:
        action_count = int(actions_npz["actions"].shape[0])
        chunk_ids = actions_npz["chunk_ids"].tolist()
        chunk_offsets = actions_npz["chunk_offsets"].tolist()
    with np.load(os.path.join(artifact_dir, "states.npz")) as states_npz:
        state_count = int(states_npz["agent_pos"].shape[0])
    trace_count = _count_jsonl(os.path.join(artifact_dir, "policy_chunk_trace.jsonl"))
    event_count = _count_jsonl(os.path.join(artifact_dir, "events.jsonl"))
    aligned = action_count == state_count == trace_count == int(result.get("num_env_steps", -1))
    return {
        "artifact_dir": os.path.abspath(artifact_dir),
        "success": bool(result.get("success", False)),
        "termination_reason": result.get("termination_reason"),
        "num_env_steps": int(result.get("num_env_steps", 0)),
        "action_count": action_count,
        "state_count": state_count,
        "trace_count": trace_count,
        "event_count": event_count,
        "aligned": bool(aligned),
        "chunk_ids": sorted(set(int(x) for x in chunk_ids)),
        "chunk_offsets": chunk_offsets[: min(10, len(chunk_offsets))],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect one Phase 4 rollout artifact directory.")
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args(argv)
    summary = inspect_rollout(args.artifact_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["aligned"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

