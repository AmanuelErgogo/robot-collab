#!/usr/bin/env python
"""Print the LLM planner's output from a run's saved prompt artifacts.

Every planner call is saved by the RoCo prompter as
``<prompt_dir>/planner_call_NNN/replan{k}_{ts}.json`` — a JSON array of
``{sender, message}`` turns (SystemPrompt, UserPrompt, then the model output:
``Planner`` for chat/plan, or per-agent turns for dialog). This walks those files
in order and prints the model output for each round, so you can see exactly what
the planner proposed.

Point it at either the prompt dir or its parent run dir:

    python scripts/show_plans.py results/equivalence/sandwich/pipeline_prompts_CRIE_BT_RR_Cent
    python scripts/show_plans.py results/equivalence/sandwich            # searches under it
    python scripts/show_plans.py <dir> --full                           # also show the prompt
    python scripts/show_plans.py <dir> --execute-only                   # only the EXECUTE block
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List

_SKIP_SENDERS = {"SystemPrompt", "UserPrompt"}


def find_round_files(root: str) -> List[str]:
    files = glob.glob(os.path.join(root, "**", "planner_call_*", "replan*.json"), recursive=True)
    files += glob.glob(os.path.join(root, "planner_call_*", "replan*.json"))
    files = sorted({f for f in files if "feedback" not in os.path.basename(f)})
    return files


def _execute_block(text: str) -> str:
    if "EXECUTE" not in text:
        return "(no EXECUTE block)"
    return "EXECUTE" + text.split("EXECUTE", 1)[1]


def show(path: str, full: bool, execute_only: bool) -> None:
    try:
        turns = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print("  (could not read {}: {})".format(path, exc))
        return
    if not isinstance(turns, list):
        return
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        sender = turn.get("sender")
        message = (turn.get("message") or "").strip()
        if not message:
            continue
        if sender in _SKIP_SENDERS and not full:
            continue
        if sender not in _SKIP_SENDERS:  # model output
            body = _execute_block(message) if execute_only else message
            print("  [{}]\n{}".format(sender, _indent(body)))
        elif full:  # prompt, only with --full
            print("  [{}]\n{}".format(sender, _indent(message)))


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A prompt dir or a run dir that contains one.")
    parser.add_argument("--full", action="store_true", help="Also print the system/user prompt.")
    parser.add_argument("--execute-only", action="store_true", help="Print only the EXECUTE block from each output.")
    args = parser.parse_args(argv)

    files = find_round_files(args.run_dir)
    if not files:
        print("No planner artifacts (planner_call_*/replan*.json) found under {}".format(args.run_dir))
        return 1
    for path in files:
        call = os.path.basename(os.path.dirname(path))
        rnd = os.path.splitext(os.path.basename(path))[0]
        print("\n=== {} / {} ===".format(call, rnd))
        show(path, args.full, args.execute_only)
    print("\n{} planner round(s).".format(len(files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
