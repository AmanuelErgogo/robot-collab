#!/usr/bin/env python
"""Run the Phase 8 benchmark release demo end to end.

The demo intentionally spans both runtime boundaries:

- the RoCo Python 3.8 simulator environment starts the Phase 0 bridge and runs
  the real RRT expert baseline;
- the LeRobot/Gym client environment validates bridge reset/step/render and
  runs the hold baseline through the benchmark evaluator.
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("{} did not contain a JSON object".format(path))
    return data


def _conda_env_exists(name: str) -> bool:
    conda = shutil.which("conda")
    if conda is None:
        return False
    try:
        proc = subprocess.run(
            [conda, "env", "list"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    return any(line.split()[0] == name for line in proc.stdout.splitlines() if line.strip() and not line.startswith("#"))


def _default_python_command(env_var: str, conda_env: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    if _conda_env_exists(conda_env):
        return "conda run -n {} python".format(conda_env)
    return sys.executable


def _split_command(command: str) -> List[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("empty command")
    return parts


def _run_step(
    name: str,
    command: Sequence[str],
    output_dir: str,
    timeout_s: int,
    check: bool = True,
) -> Dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - start
    log_path = os.path.join(output_dir, "logs", "{}.log".format(name))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    record = {
        "name": name,
        "command": list(command),
        "exit_code": int(proc.returncode),
        "elapsed_s": elapsed,
        "log_path": log_path,
    }
    if check and proc.returncode != 0:
        raise RuntimeError("{} failed with exit code {}. See {}".format(name, proc.returncode, log_path))
    return record


def _start_bridge(
    roco_python: Sequence[str],
    endpoint: str,
    output_dir: str,
    max_episode_steps: int,
) -> subprocess.Popen:
    log_path = os.path.join(output_dir, "logs", "bridge_server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    command = list(roco_python) + [
        "scripts/start_roco_bridge.py",
        "--active-agent",
        "Alice",
        "--endpoint",
        endpoint,
        "--seed",
        "0",
        "--image-height",
        "64",
        "--image-width",
        "64",
        "--max-episode-steps",
        str(max_episode_steps),
        "--headless",
    ]
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _stop_bridge(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def _wait_for_bridge(
    client_python: Sequence[str],
    endpoint: str,
    output_dir: str,
    attempts: int,
    timeout_s: int,
) -> Dict[str, Any]:
    last_error = None
    output_path = os.path.join(output_dir, "release_validation_live_bridge.json")
    for attempt in range(1, attempts + 1):
        try:
            return _run_step(
                "release_validation_live_bridge",
                list(client_python)
                + [
                    "scripts/validate_rocobench_release.py",
                    "--run-bridge-smoke",
                    "--require-bridge",
                    "--output",
                    output_path,
                ],
                output_dir,
                timeout_s=timeout_s,
                check=True,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError("bridge did not become ready: {}".format(last_error))


def _load_metrics(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        return _read_json(path)
    return {}


def _write_markdown_summary(path: str, summary: Dict[str, Any]) -> None:
    lines = [
        "# Phase 8 Demo Summary",
        "",
        "- output_root: `{}`".format(summary["output_root"]),
        "- variation_manifest_hash: `{}`".format(summary.get("variation_manifest_hash", "")),
        "- hold_success_rate: `{}`".format(summary.get("hold", {}).get("metrics", {}).get("success_rate")),
        "- rrt_success_rate: `{}`".format(summary.get("rrt", {}).get("metrics", {}).get("success_rate")),
        "- paired_n: `{}`".format(summary.get("comparison", {}).get("paired_n")),
        "",
        "## Artifacts",
        "",
    ]
    for key in ("tier1_validation", "live_bridge_validation", "hold", "rrt", "comparison"):
        value = summary.get(key)
        if not value:
            continue
        if isinstance(value, dict) and "path" in value:
            lines.append("- {}: `{}`".format(key, value["path"]))
        elif isinstance(value, dict) and "run_dir" in value:
            lines.append("- {}: `{}`".format(key, value["run_dir"]))
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = os.path.abspath(args.output_root)
    if args.overwrite and os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    roco_python = _split_command(args.roco_python)
    client_python = _split_command(args.client_python)
    current_python = _split_command(args.current_python)
    steps: List[Dict[str, Any]] = []
    bridge_proc: Optional[subprocess.Popen] = None

    tier1_path = os.path.join(output_root, "release_validation.json")
    steps.append(
        _run_step(
            "release_validation_tier1",
            current_python + ["scripts/validate_rocobench_release.py", "--output", tier1_path],
            output_root,
            timeout_s=args.command_timeout_s,
        )
    )

    if not args.skip_bridge:
        bridge_proc = _start_bridge(roco_python, args.endpoint, output_root, args.max_episode_steps)
        try:
            steps.append(
                _wait_for_bridge(
                    client_python,
                    args.endpoint,
                    output_root,
                    attempts=args.bridge_attempts,
                    timeout_s=args.command_timeout_s,
                )
            )
            hold_dir = os.path.join(output_root, "hold_bridge")
            steps.append(
                _run_step(
                    "hold_bridge_benchmark",
                    client_python
                    + [
                        "scripts/evaluate_rocobench.py",
                        "--track",
                        "skill_policy",
                        "--method",
                        "hold",
                        "--task-id",
                        args.task_id,
                        "--episodes-per-task",
                        str(args.episodes_per_task),
                        "--output",
                        hold_dir,
                        "--endpoint",
                        args.endpoint,
                        "--overwrite",
                    ],
                    output_root,
                    timeout_s=args.command_timeout_s,
                )
            )
        finally:
            _stop_bridge(bridge_proc)

    rrt_dir = os.path.join(output_root, "rrt")
    if not args.skip_rrt:
        steps.append(
            _run_step(
                "rrt_benchmark",
                roco_python
                + [
                    "scripts/evaluate_rocobench.py",
                    "--track",
                    "skill_policy",
                    "--method",
                    "rrt",
                    "--task-id",
                    args.task_id,
                    "--episodes-per-task",
                    str(args.episodes_per_task),
                    "--output",
                    rrt_dir,
                    "--overwrite",
                ],
                output_root,
                timeout_s=args.rrt_timeout_s,
            )
        )

    comparison_path = os.path.join(output_root, "rrt_vs_hold_compare.json")
    if not args.skip_bridge and not args.skip_rrt and not args.skip_compare:
        steps.append(
            _run_step(
                "paired_comparison",
                current_python
                + [
                    "scripts/compare_rocobench_runs.py",
                    rrt_dir,
                    os.path.join(output_root, "hold_bridge"),
                    "--output",
                    comparison_path,
                ],
                output_root,
                timeout_s=args.command_timeout_s,
            )
        )

    summary: Dict[str, Any] = {
        "output_root": output_root,
        "task_id": args.task_id,
        "episodes_per_task": int(args.episodes_per_task),
        "commands": steps,
        "tier1_validation": {"path": tier1_path, "result": _read_json(tier1_path)},
    }
    live_path = os.path.join(output_root, "release_validation_live_bridge.json")
    if os.path.exists(live_path):
        summary["live_bridge_validation"] = {"path": live_path, "result": _read_json(live_path)}
    hold_dir = os.path.join(output_root, "hold_bridge")
    if os.path.exists(hold_dir):
        summary["hold"] = {"run_dir": hold_dir, "metrics": _load_metrics(hold_dir)}
    if os.path.exists(rrt_dir):
        summary["rrt"] = {"run_dir": rrt_dir, "metrics": _load_metrics(rrt_dir)}
    if os.path.exists(comparison_path):
        comparison = _read_json(comparison_path)
        summary["comparison"] = comparison
    rrt_manifest = os.path.join(rrt_dir, "run_manifest.json")
    hold_manifest = os.path.join(hold_dir, "run_manifest.json")
    for manifest_path in (rrt_manifest, hold_manifest):
        if os.path.exists(manifest_path):
            summary["variation_manifest_hash"] = _read_json(manifest_path).get("variation_manifest_hash")
            break

    summary_path = os.path.join(output_root, "demo_summary.json")
    _write_json(summary_path, summary)
    _write_markdown_summary(os.path.join(output_root, "demo_summary.md"), summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/benchmark/phase8_demo")
    parser.add_argument("--task-id", default="pack.put.alice")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--max-episode-steps", type=int, default=4)
    parser.add_argument("--roco-python", default=_default_python_command("ROCO_PYTHON_BIN", "roco"))
    parser.add_argument("--client-python", default=_default_python_command("LEROBOT_ROCO_PYTHON_BIN", "lerobot-roco"))
    parser.add_argument("--current-python", default=sys.executable)
    parser.add_argument("--command-timeout-s", type=int, default=120)
    parser.add_argument("--rrt-timeout-s", type=int, default=360)
    parser.add_argument("--bridge-attempts", type=int, default=8)
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--skip-rrt", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    summary = run_demo(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

