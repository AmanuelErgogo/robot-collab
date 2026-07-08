#!/usr/bin/env python
"""Equivalence check: legacy runner vs new pipeline on the same task/seed.

The merge (docs/crie-bt/crie_bt_pipeline_merge.md) makes the new registry-driven
pipeline run the *same* real engine (LLM planner + RRT executor + coded monitor)
as the legacy ``scripts/run_crie_bt_sim.py``. This script runs both paths on the
same task and seed and diffs the comparable metrics (task success, steps,
replans, local retries, planner calls), so you can confirm the merge did not
change behaviour.

Requires an offscreen GL backend (``MUJOCO_GL=egl``). Two input modes:

* real LLM (the meaningful check) -- pass ``--llm-source gemini-2.5-flash`` (+ the
  credentials in docs/getting-started/llm_setup_and_credentials.md);
* no-LLM smoke -- omit ``--llm-source`` to drive both paths with a WAIT action
  (validates the harness headlessly; metrics are trivial and the loop-termination
  on a non-completing action may legitimately differ).

Condition -> legacy flags:
    CRIE-BT-RR-Cent   -> --mode bt_mediated              --planner-mode chat
    CRIE-BT-RR-Dialog -> --mode bt_mediated              --planner-mode dialog
    VLM-RR-Cent       -> --mode vlm_sarm_monitor_planner --planner-mode chat
    VLM-RR-Dialog     -> --mode vlm_sarm_monitor_planner --planner-mode dialog

Examples::

    MUJOCO_GL=egl python scripts/equivalence_check.py --task sandwich --seed 0 \
        --conditions CRIE-BT-RR-Dialog VLM-RR-Dialog --llm-source gemini-2.5-flash

    # headless harness smoke (no credentials needed):
    MUJOCO_GL=egl python scripts/equivalence_check.py --task pack --no-llm

    # compare two already-produced jsonl rows:
    python scripts/equivalence_check.py --compare legacy.jsonl pipeline.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONBREAKPOINT", "0")

CONDITION_TO_LEGACY = {
    "CRIE-BT-RR-Cent": ("bt_mediated", "chat"),
    "CRIE-BT-RR-Dialog": ("bt_mediated", "dialog"),
    "VLM-RR-Cent": ("vlm_sarm_monitor_planner", "chat"),
    "VLM-RR-Dialog": ("vlm_sarm_monitor_planner", "dialog"),
}

# Metrics that both paths report and that should agree if behaviour is preserved.
COMPARE_KEYS = ["success", "steps", "replans", "local_retries", "planner_calls"]


def _read_last_row(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise RuntimeError("no rows written to {}".format(path))
    return rows[-1]


def normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a legacy or pipeline row to the comparable metric set."""
    def get(*keys):
        for k in keys:
            if row.get(k) is not None:
                return row[k]
        return None

    # Task-level success: legacy logs it as sim_success; the pipeline's success
    # is already env.get_task_done().
    success = row["sim_success"] if "sim_success" in row else get("success")
    return {
        "success": bool(success) if success is not None else None,
        "steps": get("num_steps", "steps"),
        "replans": get("replans"),
        "local_retries": get("local_retries"),
        "planner_calls": get("planner_calls"),
    }


def run_legacy(task, seed, condition, llm_source, max_steps, out_path, no_llm, video_dir=None) -> Dict[str, Any]:
    mode, planner_mode = CONDITION_TO_LEGACY[condition]
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "run_crie_bt_sim.py"),
           "--task", task, "--adapter", "legacy", "--seed", str(seed),
           "--episodes", "1", "--max-steps", str(max_steps), "--output", out_path, "--mode", mode]
    if no_llm:
        cmd += ["--planner-mode", "legacy_action"]  # emits WAIT, no LLM
    else:
        cmd += ["--planner-mode", planner_mode, "--llm-source", llm_source]
    if video_dir:
        cmd += ["--artifact-dir", video_dir]  # RRT executor writes execute.mp4 here
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return _read_last_row(out_path)


def run_pipeline(task, seed, condition, llm_source, max_steps, out_path, no_llm, video_dir=None) -> Dict[str, Any]:
    from rocobench.crie_bt.pipeline import build_condition

    cfg = build_condition(condition, "step1")
    if no_llm:
        # No-LLM: real env + RRT executor + coded monitor driven by a WAIT action.
        from rocobench.crie_bt.pipeline import build_collaboration_controller
        from rocobench.crie_bt.pipeline.roco_backend import build_roco_step1

        env_adapter, executor, planner_factory = build_roco_step1(task, seed=seed, artifact_dir=video_dir)
        controller = build_collaboration_controller(cfg, planner_factory=planner_factory, executor=executor)
    else:
        from rocobench.crie_bt.pipeline.roco_backend import build_roco_condition

        prompt_dir = os.path.join(os.path.dirname(out_path), "pipeline_prompts_" + condition.replace("-", "_"))
        controller, env_adapter = build_roco_condition(
            cfg, task, seed=seed, llm_source=llm_source, prompt_save_dir=prompt_dir, artifact_dir=video_dir)
    row = controller.run_episode(condition, env_adapter, task, task, max_steps, seed=seed)
    row.pop("events", None)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def compare(legacy_row: Dict[str, Any], pipeline_row: Dict[str, Any], condition: str) -> Dict[str, Any]:
    lg, pl = normalize(legacy_row), normalize(pipeline_row)
    diffs = {k: (lg[k], pl[k]) for k in COMPARE_KEYS if lg[k] != pl[k]}
    return {"condition": condition, "legacy": lg, "pipeline": pl,
            "match": not diffs, "diffs": diffs}


def _print_report(results: List[Dict[str, Any]]) -> bool:
    all_match = True
    for res in results:
        cond, lg, pl = res["condition"], res["legacy"], res["pipeline"]
        mark = "✅ match" if res["match"] else "⚠️  diverge"
        print("\n== {} : {} ==".format(cond, mark))
        print("  {:<15}{:>12}{:>12}".format("metric", "legacy", "pipeline"))
        for k in COMPARE_KEYS:
            flag = "" if lg[k] == pl[k] else "   <-- diff"
            print("  {:<15}{:>12}{:>12}{}".format(k, str(lg[k]), str(pl[k]), flag))
        all_match = all_match and res["match"]
    print("\nOVERALL:", "✅ all conditions match" if all_match else "⚠️  divergence found")
    return all_match


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar=("LEGACY_JSONL", "PIPELINE_JSONL"), default=None,
                        help="Compare two already-produced result rows and exit.")
    parser.add_argument("--task", default="pack")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="*", default=["CRIE-BT-RR-Cent"],
                        choices=list(CONDITION_TO_LEGACY.keys()))
    parser.add_argument("--llm-source", default=None, help="e.g. gemini-2.5-flash (omit for --no-llm).")
    parser.add_argument("--no-llm", action="store_true", help="Drive both paths with a WAIT action (headless smoke).")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--record-video", action="store_true",
                        help="Record execute.mp4 per path under --out-dir (renders the task's teaser camera).")
    parser.add_argument("--out-dir", default="results/equivalence")
    args = parser.parse_args(argv)

    if args.compare:
        res = compare(_read_last_row(args.compare[0]), _read_last_row(args.compare[1]), "supplied-rows")
        return 0 if _print_report([res]) else 1

    no_llm = args.no_llm or not args.llm_source
    llm_source = args.llm_source or "gpt-4"
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for condition in args.conditions:
        tag = "{}_{}_seed{}".format(condition.replace("-", "_"), args.task, args.seed)
        legacy_out = os.path.join(out_dir, "legacy_" + tag + ".jsonl")
        pipeline_out = os.path.join(out_dir, "pipeline_" + tag + ".jsonl")
        legacy_vid = os.path.join(out_dir, "legacy_video_" + tag) if args.record_video else None
        pipeline_vid = os.path.join(out_dir, "pipeline_video_" + tag) if args.record_video else None
        print("[legacy]   {} ...".format(condition))
        legacy_row = run_legacy(args.task, args.seed, condition, llm_source, args.max_steps,
                                legacy_out, no_llm, video_dir=legacy_vid)
        print("[pipeline] {} ...".format(condition))
        pipeline_row = run_pipeline(args.task, args.seed, condition, llm_source, args.max_steps,
                                    pipeline_out, no_llm, video_dir=pipeline_vid)
        if args.record_video:
            print("           video: {}/execute.mp4  |  {}/execute.mp4".format(legacy_vid, pipeline_vid))
        results.append(compare(legacy_row, pipeline_row, condition))

    if no_llm:
        print("\n(NOTE: --no-llm WAIT smoke — validates the harness only. Run with "
              "--llm-source for a meaningful equivalence check.)")
    ok = _print_report(results)
    return 0 if ok or no_llm else 1


if __name__ == "__main__":
    raise SystemExit(main())
