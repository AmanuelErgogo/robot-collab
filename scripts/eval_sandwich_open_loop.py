#!/usr/bin/env python
"""Evaluate plan / chat / dialog + open_loop on the sandwich task.

Runs N episodes of each LLM communication mode against the MuJoCo sandwich
simulator and reports the full set of metrics needed for paper tables:

  Task Success Rate (SR)   — env.get_reward_done() is True at episode end
  Planner Success Rate     — CRIE-BT internal step completion rate
  Avg Steps to Done        — simulator steps when SR=1, else max_steps
  LLM Calls / Episode      — how many prompter calls were made
  Avg Wall Time (s)        — wall clock seconds per episode
  Avg LLM Latency (s)      — per individual LLM API call
  Avg Tokens / Episode     — prompt + completion tokens (when available)
  Replans / Episode        — should always be 0 for open_loop
  Failure Distribution     — breakdown of FailureCode counts

Use --dry-run to run with fake components (no MuJoCo / no API key) for CI or
quick sanity checks.  All results are written to --output-dir as:
  episodes.jsonl   — one row per episode
  summary.csv      — aggregated per-mode metrics
  table.txt        — ASCII table (copy-paste to paper draft)
  table.tex        — LaTeX booktabs table (paste into paper)
  failures.json    — per-mode failure code breakdown

Usage (dry-run, no simulator):
    python scripts/eval_sandwich_open_loop.py --dry-run --episodes 5

Usage (real simulator, Gemini):
    conda run --no-capture-output -n roco env MUJOCO_GL=egl \\
      python scripts/eval_sandwich_open_loop.py \\
        --episodes 20 \\
        --seeds 0 1 2 3 4 \\
        --llm-source gemini-2.0-flash-exp \\
        --api-key-path /path/to/gemini.key \\
        --output-dir results/sandwich_open_loop/

Usage (real simulator, GPT-4o):
    conda run --no-capture-output -n roco env MUJOCO_GL=egl \\
      python scripts/eval_sandwich_open_loop.py \\
        --episodes 20 \\
        --llm-source gpt-4o \\
        --output-dir results/sandwich_open_loop/
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("PYTHONBREAKPOINT", "0")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - delta), min(1.0, centre + delta)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _fmt(val: float, decimals: int = 3) -> str:
    return "{:.{}f}".format(val, decimals)


# ---------------------------------------------------------------------------
# Prompter instrumentation — wraps any prompter to record timing + tokens
# ---------------------------------------------------------------------------

class _InstrumentedPrompter:
    """Wraps a RoCoBench prompter; records per-call timing and token usage."""

    def __init__(self, prompter: Any) -> None:
        self._prompter = prompter
        self.call_times: List[float] = []       # wall seconds per call
        self.call_tokens: List[int] = []        # tokens per call (0 if unknown)
        self.total_calls: int = 0

    def prompt_one_round(self, obs: Any, save_path: str = "") -> Any:
        t0 = time.perf_counter()
        result = self._prompter.prompt_one_round(obs, save_path=save_path)
        elapsed = time.perf_counter() - t0
        self.call_times.append(elapsed)
        self.total_calls += 1
        # Try to extract token counts from prompter state (varies by implementation)
        tokens = _extract_token_count(self._prompter)
        self.call_tokens.append(tokens)
        return result

    # Forward unknown attributes to the wrapped prompter (e.g. query_once)
    def __getattr__(self, name: str) -> Any:
        return getattr(self._prompter, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in ("call_times", "call_tokens", "total_calls"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._prompter, name)


def _extract_token_count(prompter: Any) -> int:
    """Try to pull token count from prompter internal state."""
    for attr in ("total_tokens", "_total_tokens", "token_count"):
        val = getattr(prompter, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return 0


# ---------------------------------------------------------------------------
# Fake components for --dry-run mode (no MuJoCo, no LLM key)
# ---------------------------------------------------------------------------

SANDWICH_AGENTS = ["Chad", "Dave"]
SANDWICH_CONTEXT = (
    "2 robots, Chad and Dave, together make a [bacon_sandwich].\n"
    "Recipe order: bread_slice1 → bacon → cheese → tomato → bread_slice2\n"
    "bread_slice1 must be PUT on cutting_board first.\n"
    "Chad can only reach the right side; Dave can only reach the left side."
)
SANDWICH_ACTION_PROMPT = (
    "[Action Options]\n"
    "1) PICK <obj>\n2) PUT <obj1> <obj2>\n3) WAIT\n"
    "Output: EXECUTE\\nNAME <agent> ACTION <action>"
)

# Scripted recipe for the fake sim — one EXECUTE per round
_RECIPE = [
    "EXECUTE\nNAME Dave ACTION PICK bread_slice1\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PUT bread_slice1 cutting_board\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Chad ACTION PICK bacon\nNAME Dave ACTION WAIT",
    "EXECUTE\nNAME Chad ACTION PUT bacon bread_slice1\nNAME Dave ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PICK cheese\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PUT cheese bacon\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Chad ACTION PICK tomato\nNAME Dave ACTION WAIT",
    "EXECUTE\nNAME Chad ACTION PUT tomato cheese\nNAME Dave ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PICK bread_slice2\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PUT bread_slice2 tomato\nNAME Chad ACTION WAIT",
]


class _FakeSandwichEnv:
    robot_name_map = {"ur5e_robotiq": "Chad", "panda": "Dave"}

    def __init__(self, success_after: int = 1, seed: int = 0):
        """Succeed after `success_after` execute steps."""
        self._success_after = max(1, success_after)
        self._seed = seed
        self._steps = 0
        self._done = False

    def reset(self):
        self._steps = 0
        self._done = False
        return self._obs()

    def get_obs(self):
        return self._obs()

    def get_action_prompt(self):
        return SANDWICH_ACTION_PROMPT

    def describe_task_context(self):
        return SANDWICH_CONTEXT

    def get_task_feedback(self, path_plan, pose_dict=None):
        return ""

    def get_reward_done(self, obs):
        return (1.0 if self._done else 0.0), self._done

    def describe_obs(self, obs):
        return "[fake obs step={}]".format(self._steps)

    def seed(self, np_seed=0):
        pass

    def _obs(self):
        class _O:
            done = self._done
            steps = self._steps
        return _O()


class _FakeParser:
    def parse(self, obs, response):
        return True, "", ["compiled"]


class _FakeSimExecutor:
    """Marks env as done after success_after calls."""

    def __init__(self, env: _FakeSandwichEnv, latency_s: float = 0.01):
        self.env = env
        self.latency = latency_s
        self.calls = 0

    def execute(self, plan, obs, artifact_dir=None):
        from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus
        time.sleep(self.latency)  # simulate execution time
        self.calls += 1
        self.env._steps += 1
        if self.env._steps >= self.env._success_after:
            self.env._done = True
        return SkillExecutionResult(
            success=True,
            status=SkillExecutionStatus.SUCCESS,
            reason="",
            num_sim_steps=30,
            reward=1.0 if self.env._done else 0.0,
            done=self.env._done,
            info={},
        )


class _FakeLLMPathPlan:
    def __init__(self, text: str):
        self.parsed_proposal = text


class _FakeModePrompter:
    """Returns scripted EXECUTE blocks; simulates small LLM latency."""

    def __init__(self, mode: str, latency_s: float = 0.02, fail_rate: float = 0.0, seed: int = 0):
        self.mode = mode
        self.latency = latency_s
        self.fail_rate = fail_rate
        self._calls = 0
        import random
        self._rng = random.Random(seed)

    def prompt_one_round(self, obs, save_path: str = ""):
        time.sleep(self.latency)
        self._calls += 1
        if self._rng.random() < self.fail_rate:
            return False, [], [], []
        # dialog mode simulates extra calls (one per agent)
        if self.mode == "dialog":
            time.sleep(self.latency)  # extra per-agent call cost
        idx = (self._calls - 1) % len(_RECIPE)
        response = _RECIPE[idx]
        plan = _FakeLLMPathPlan(response)
        return True, [plan], ["ok"], [response]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _describe_obs(env: Any, obs: Any) -> str:
    if obs is not None and hasattr(env, "describe_obs"):
        return env.describe_obs(obs)
    return ""


def _sim_success(env: Any, obs: Any, fallback: bool) -> bool:
    if obs is not None and hasattr(env, "get_reward_done"):
        try:
            return bool(env.get_reward_done(obs)[1])
        except Exception:
            pass
    return bool(fallback)


def run_one_episode(
    env: Any,
    planner_mode: str,
    prompter: Any,
    args: argparse.Namespace,
    episode: int,
    seed: int,
) -> Dict[str, Any]:
    """Build adapter+controller, run one episode, return annotated log row."""
    from rocobench.crie_bt.controllers import build_controller
    from rocobench.crie_bt.legacy_tasks import (
        LegacyPromptPlanner,
        LegacyTaskRRTExecutorAdapter,
        agent_names_for_env,
        legacy_task_spec,
    )

    agent_names = agent_names_for_env(env)
    instrumented = _InstrumentedPrompter(prompter)
    planner = LegacyPromptPlanner(
        task_id="sandwich",
        prompter=instrumented,
        planner_mode=planner_mode,
        agent_names=agent_names,
        save_dir=_prompt_save_dir(args, planner_mode, episode, seed),
    )

    if args.dry_run:
        sim_executor = _FakeSimExecutor(env, latency_s=0.005)
        parser = _FakeParser()
    else:
        from prompting.parser import LLMResponseParser
        from rocobench.skills import RRTSkillExecutor
        parser = LLMResponseParser(
            env,
            "action_only",
            env.robot_name_map,
            ["NAME", "ACTION"],
            use_prepick=getattr(env, "use_prepick", False),
            use_preplace=getattr(env, "use_preplace", False),
        )
        sim_executor = RRTSkillExecutor(
            env=env,
            robots=env.get_sim_robots(),
            max_sim_steps=int(args.max_sim_steps),
        )

    from rocobench.crie_bt.legacy_tasks import LegacyTaskRRTExecutorAdapter
    adapter = LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id="sandwich",
        parser=parser,
        executor=sim_executor,
        artifact_dir=_artifact_dir(args, planner_mode, episode, seed),
        uncertainty_reporter=None,
    )

    controller = build_controller("open_loop", planner, adapter, uncertainty_mode=args.uncertainty)

    obs_before = env.get_obs() if hasattr(env, "get_obs") else None
    t_start = time.perf_counter()
    log = controller.run_episode(env, args.task_goal, int(args.max_steps))
    wall_time = time.perf_counter() - t_start
    obs_after = env.get_obs() if hasattr(env, "get_obs") else None

    # Annotate with eval metadata
    log["episode"] = episode
    log["seed"] = seed
    log["planner_mode"] = planner_mode
    log["task_id"] = "sandwich"
    log["task_name"] = env.__class__.__name__
    log["adapter"] = "legacy"
    log["sim_success"] = _sim_success(env, obs_after, log.get("success", False))
    log["initial_scene"] = _describe_obs(env, obs_before)
    log["final_scene"] = _describe_obs(env, obs_after)
    log["wall_time_s"] = round(wall_time, 4)
    log["llm_calls"] = instrumented.total_calls
    log["llm_call_times_s"] = [round(t, 4) for t in instrumented.call_times]
    log["llm_avg_latency_s"] = round(_mean(instrumented.call_times), 4)
    log["llm_total_tokens"] = sum(instrumented.call_tokens)
    log["task_spec"] = _json_safe(legacy_task_spec(env, "sandwich"))

    return log


def _prompt_save_dir(args: argparse.Namespace, planner_mode: str, episode: int, seed: int) -> str:
    if not args.output_dir:
        return ""
    return os.path.join(args.output_dir, "prompts", planner_mode, "ep{:03d}_s{:02d}".format(episode, seed))


def _artifact_dir(args: argparse.Namespace, planner_mode: str, episode: int, seed: int) -> Optional[str]:
    if not args.output_dir or not args.save_artifacts:
        return None
    d = os.path.join(args.output_dir, "artifacts", planner_mode, "ep{:03d}_s{:02d}".format(episode, seed))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Environment / prompter builders
# ---------------------------------------------------------------------------

def _build_env(seed: int, args: argparse.Namespace) -> Any:
    if args.dry_run:
        env = _FakeSandwichEnv(success_after=1, seed=seed)
        env.reset()
        return env
    from rocobench.crie_bt.legacy_tasks import make_legacy_task_env
    return make_legacy_task_env("sandwich", seed=seed)


def _build_prompter(planner_mode: str, env: Any, args: argparse.Namespace, seed: int) -> Any:
    if args.dry_run:
        return _FakeModePrompter(planner_mode, latency_s=0.02, fail_rate=0.0, seed=seed)
    return _build_real_prompter(planner_mode, env, args)


def _build_real_prompter(planner_mode: str, env: Any, args: argparse.Namespace) -> Any:
    from prompting import DialogPrompter, FeedbackManager, LLMResponseParser, SingleThreadPrompter
    from rocobench import MultiArmRRT

    parser = LLMResponseParser(
        env,
        "action_only",
        env.robot_name_map,
        ["NAME", "ACTION"],
        0,
        use_prepick=getattr(env, "use_prepick", False),
        use_preplace=getattr(env, "use_preplace", False),
        split_parsed_plans=False,
    )
    rrt_planner = MultiArmRRT(
        env.physics,
        robots=env.get_sim_robots(),
        graspable_object_names=env.get_graspable_objects(),
        allowed_collision_pairs=env.get_allowed_collision_pairs(),
    )
    feedback_manager = FeedbackManager(
        env=env,
        planner=rrt_planner,
        llm_output_mode="action_only",
        robot_name_map=env.robot_name_map,
        step_std_threshold=getattr(env, "waypoint_std_threshold", 0.1),
        max_failed_waypoints=1,
    )
    if planner_mode in ("plan", "chat"):
        prompter = SingleThreadPrompter(
            env=env,
            parser=parser,
            feedback_manager=feedback_manager,
            max_tokens=int(args.max_tokens),
            debug_mode=False,
            use_waypoints=False,
            use_history=(planner_mode == "chat"),
            num_replans=int(args.num_replans),
            comm_mode=planner_mode,
            temperature=float(args.temperature),
            llm_source=_prompter_llm_source(args),
        )
    else:  # dialog
        prompter = DialogPrompter(
            env=env,
            parser=parser,
            feedback_manager=feedback_manager,
            max_tokens=int(args.max_tokens),
            debug_mode=False,
            robot_name_map=env.robot_name_map,
            max_calls_per_round=int(args.max_calls_per_round),
            use_waypoints=False,
            use_history=True,
            use_feedback=True,
            num_replans=int(args.num_replans),
            temperature=float(args.temperature),
            llm_source=_prompter_llm_source(args),
        )
    if _uses_llm_api(args):
        _install_bridge(prompter, args)
    return prompter


def _uses_llm_api(args: argparse.Namespace) -> bool:
    return str(args.llm_source).strip().lower().startswith(("gemini", "claude", "llama"))


def _prompter_llm_source(args: argparse.Namespace) -> str:
    return "gpt-4" if _uses_llm_api(args) else args.llm_source


def _install_bridge(prompter: Any, args: argparse.Namespace) -> None:
    from llm_api import create_llm_client
    client = create_llm_client(args.llm_source, api_key_path=(args.api_key_path or None))

    total_tokens = [0]

    def query_once(system_prompt, user_prompt="", max_query=None):
        del max_query
        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        response = client.generate(
            messages=messages,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
        )
        if hasattr(response, "usage") and response.usage:
            try:
                total_tokens[0] += int(response.usage.get("total_tokens", 0))
            except (TypeError, AttributeError):
                pass
        return response.text, response.usage

    prompter.query_once = query_once
    prompter._total_tokens = total_tokens[0]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> List[Dict[str, Any]]:
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "episodes.jsonl")
    modes = args.modes
    seeds = args.seeds
    total = len(modes) * len(seeds) * args.episodes
    done = 0
    rows: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for planner_mode in modes:
            for seed in seeds:
                for ep in range(args.episodes):
                    done += 1
                    print("[{:3d}/{:3d}]  mode={:<8s}  seed={:2d}  episode={:3d}  ...".format(
                        done, total, planner_mode, seed, ep), end="", flush=True)
                    env = _build_env(seed + ep, args)
                    prompter = _build_prompter(planner_mode, env, args, seed + ep)
                    try:
                        row = run_one_episode(env, planner_mode, prompter, args, ep, seed)
                        status = "SR={} t={:.1f}s".format(
                            "✓" if row["sim_success"] else "✗", row["wall_time_s"])
                    except Exception as exc:
                        row = _error_row(planner_mode, ep, seed, exc)
                        status = "ERROR: {}".format(str(exc)[:60])
                    print("  " + status)
                    f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
                    f.flush()
                    rows.append(row)
    return rows


def _error_row(planner_mode: str, ep: int, seed: int, exc: Exception) -> Dict[str, Any]:
    return {
        "episode": ep, "seed": seed, "planner_mode": planner_mode,
        "task_id": "sandwich", "mode": "open_loop",
        "success": False, "sim_success": False,
        "steps": 0, "planner_calls": 0, "replans": 0, "local_retries": 0,
        "completed_subtasks": 0, "failed_subtasks": 0,
        "failure_counts": {"RUNTIME_ERROR": 1},
        "events": [], "subtask_results": [], "explanations": [str(exc)],
        "wall_time_s": 0.0, "llm_calls": 0, "llm_avg_latency_s": 0.0,
        "llm_total_tokens": 0, "llm_call_times_s": [], "error": str(exc),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Dict[str, Any]]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    for row in rows:
        m = row.get("planner_mode", "")
        if m in by_mode:
            by_mode[m].append(row)

    stats: Dict[str, Dict[str, Any]] = {}
    for mode, items in by_mode.items():
        n = len(items)
        if n == 0:
            continue
        # Action Success Rate: LLM plan parsed + RRT executed (primary metric for open-loop)
        asr_count = sum(1 for r in items if r.get("success"))
        asr_lo, asr_hi = _wilson_ci(asr_count, n)
        # Task Completion Rate: env.get_reward_done() — needs multi-step execution
        tcr_count = sum(1 for r in items if r.get("sim_success"))
        tcr_lo, tcr_hi = _wilson_ci(tcr_count, n)

        all_steps = [r["steps"] for r in items]
        llm_calls = [r.get("llm_calls", 0) for r in items]
        wall_times = [r.get("wall_time_s", 0.0) for r in items]
        latencies = [r.get("llm_avg_latency_s", 0.0) for r in items]
        tokens = [r.get("llm_total_tokens", 0) for r in items]
        replans = [r.get("replans", 0) for r in items]
        planner_errors = sum(1 for r in items if "PLANNER_ERROR" in r.get("failure_counts", {}))

        failure_counts: Dict[str, int] = defaultdict(int)
        for r in items:
            for code, cnt in dict(r.get("failure_counts", {})).items():
                failure_counts[code] += int(cnt)

        stats[mode] = {
            "n": n,
            # Primary metric for open-loop mode comparison
            "action_success_count": asr_count,
            "action_success_rate": asr_count / n,
            "asr_ci_lo": asr_lo,
            "asr_ci_hi": asr_hi,
            # Task completion (requires multi-step; always ~0 for open-loop)
            "task_completion_count": tcr_count,
            "task_completion_rate": tcr_count / n,
            "tcr_ci_lo": tcr_lo,
            "tcr_ci_hi": tcr_hi,
            # Legacy alias for backward compat
            "sim_success_rate": tcr_count / n,
            "crie_success_rate": asr_count / n,
            "sr_ci_lo": asr_lo,
            "sr_ci_hi": asr_hi,
            # Efficiency
            "avg_steps_all": _mean(all_steps),
            "std_steps_all": _std(all_steps),
            "avg_steps_succ": _mean([r["steps"] for r in items if r.get("success")]) or float("nan"),
            "avg_llm_calls": _mean(llm_calls),
            "std_llm_calls": _std(llm_calls),
            "avg_wall_time_s": _mean(wall_times),
            "std_wall_time_s": _std(wall_times),
            "avg_llm_latency_s": _mean([l for l in latencies if l > 0]),
            "avg_tokens": _mean([t for t in tokens if t > 0]),
            "avg_replans": _mean(replans),
            "planner_errors": planner_errors,
            "failure_counts": dict(failure_counts),
        }
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_PAPER_MODE_LABELS = {
    "plan": "Plan (Centralised)",
    "chat": "Chat (w/ History)",
    "dialog": "Dialog (Multi-agent)",
}


def _ascii_table(stats: Dict[str, Dict[str, Any]], modes: List[str]) -> str:
    lines = []
    sep = "+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 13 + "+" + "-" * 12 + "+" + "-" * 10 + "+"
    header = (
        "| {:<20s} | {:>8s} | {:>12s} | {:>10s} | {:>11s} | {:>10s} | {:>8s} |".format(
            "Mode", "ASR (%)", "ASR 95% CI", "Steps", "Time (s)", "LLM Lat(s)", "P.Err"
        )
    )
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    n = 0
    for mode in modes:
        if mode not in stats:
            continue
        s = stats[mode]
        n = s["n"]
        asr_pct = s["action_success_rate"] * 100
        ci_str = "[{:.1f},{:.1f}]".format(s["asr_ci_lo"] * 100, s["asr_ci_hi"] * 100)
        steps = "{:.1f}±{:.1f}".format(s["avg_steps_all"], s["std_steps_all"])
        t_str = "{:.1f}±{:.1f}".format(s["avg_wall_time_s"], s["std_wall_time_s"])
        lat = "{:.1f}".format(s["avg_llm_latency_s"])
        perr = "{}/{}".format(s["planner_errors"], n)
        label = _PAPER_MODE_LABELS.get(mode, mode)
        lines.append("| {:<20s} | {:>8.1f} | {:>12s} | {:>10s} | {:>11s} | {:>10s} | {:>8s} |".format(
            label, asr_pct, ci_str, steps, t_str, lat, perr))
    lines.append(sep)
    lines.append("ASR = Action Success Rate (LLM plan valid + RRT executed); n={} per mode".format(n))
    lines.append("Task Completion Rate (full sandwich) requires multi-step execution; always ~0% for open-loop.")
    return "\n".join(lines)


def _latex_table(stats: Dict[str, Dict[str, Any]], modes: List[str], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    \textbf{Mode} & \textbf{ASR (\%)} & \textbf{95\% CI}"
        r" & \textbf{Steps} & \textbf{Time (s)} & \textbf{LLM Lat (s)} & \textbf{P.Err} \\",
        r"    \midrule",
    ]
    for mode in modes:
        if mode not in stats:
            continue
        s = stats[mode]
        n = s["n"]
        asr_pct = s["action_success_rate"] * 100
        ci_str = "[{:.0f}, {:.0f}]".format(s["asr_ci_lo"] * 100, s["asr_ci_hi"] * 100)
        steps = "${:.1f} \\pm {:.1f}$".format(s["avg_steps_all"], s["std_steps_all"])
        t_str = "${:.1f} \\pm {:.1f}$".format(s["avg_wall_time_s"], s["std_wall_time_s"])
        lat = "{:.1f}".format(s["avg_llm_latency_s"])
        perr = "{}/{}".format(s["planner_errors"], n)
        label_str = _PAPER_MODE_LABELS.get(mode, mode)
        lines.append("    {} & {:.1f} & {} & {} & {} & {} & {} \\\\".format(
            label_str, asr_pct, ci_str, steps, t_str, lat, perr))
    n = next(iter(stats.values()))["n"] if stats else 0
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \footnotesize{ASR = Action Success Rate (LLM plan parseable \& RRT executed)."
        r" P.Err = planner errors (LLM failed to produce a valid plan)."
        r" $n=%d$ episodes per mode, seeds $\in \{0,1,2\}$, open-loop controller, Gemini 2.5 Flash.}" % n,
        r"\end{table}",
    ]
    return "\n".join(lines)


def _failure_report(stats: Dict[str, Dict[str, Any]], modes: List[str]) -> str:
    lines = ["Failure code distribution per mode:"]
    for mode in modes:
        if mode not in stats:
            continue
        fc = stats[mode].get("failure_counts", {})
        label = _PAPER_MODE_LABELS.get(mode, mode)
        if not fc:
            lines.append("  {:25s}: no failures recorded".format(label))
        else:
            total = sum(fc.values())
            parts = ["{}: {}/{} ({:.0f}%)".format(k, v, total, 100 * v / total)
                     for k, v in sorted(fc.items(), key=lambda x: -x[1])]
            lines.append("  {:25s}: {}".format(label, ", ".join(parts)))
    return "\n".join(lines)


def _extended_stats_block(stats: Dict[str, Dict[str, Any]], modes: List[str]) -> str:
    lines = ["Extended per-mode statistics:", ""]
    for mode in modes:
        if mode not in stats:
            continue
        s = stats[mode]
        label = _PAPER_MODE_LABELS.get(mode, mode)
        lines.append("  {}  (n={})".format(label, s["n"]))
        lines.append("    action_success_rate : {:.3f}  [{:.3f}, {:.3f}] 95% Wilson CI".format(
            s["action_success_rate"], s["asr_ci_lo"], s["asr_ci_hi"]))
        lines.append("    task_completion_rate: {:.3f}  (sim env done; ~0 for open-loop 1-step)".format(
            s["task_completion_rate"]))
        if not math.isnan(s["avg_steps_succ"]):
            lines.append("    avg_steps (succ)    : {:.1f}".format(s["avg_steps_succ"]))
        lines.append("    avg_steps (all)     : {:.1f} ± {:.1f}".format(
            s["avg_steps_all"], s["std_steps_all"]))
        lines.append("    avg_llm_calls       : {:.1f} ± {:.1f}".format(
            s["avg_llm_calls"], s["std_llm_calls"]))
        lines.append("    avg_wall_time_s     : {:.2f} ± {:.2f}".format(
            s["avg_wall_time_s"], s["std_wall_time_s"]))
        if s["avg_llm_latency_s"] > 0:
            lines.append("    avg_llm_latency_s   : {:.3f}".format(s["avg_llm_latency_s"]))
        if s["avg_tokens"] > 0:
            lines.append("    avg_tokens          : {:.0f}".format(s["avg_tokens"]))
        lines.append("    avg_replans         : {:.2f}".format(s["avg_replans"]))
        lines.append("    planner_errors      : {}".format(s["planner_errors"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(stats: Dict[str, Dict[str, Any]], modes: List[str], args: argparse.Namespace) -> Dict[str, str]:
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    paths: Dict[str, str] = {}

    # summary.csv
    csv_path = os.path.join(out, "summary.csv")
    fields = [
        "mode", "n", "sim_success_rate", "sr_ci_lo", "sr_ci_hi",
        "crie_success_rate", "avg_steps_all", "std_steps_all",
        "avg_llm_calls", "std_llm_calls", "avg_wall_time_s", "std_wall_time_s",
        "avg_llm_latency_s", "avg_tokens", "avg_replans", "failure_counts",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mode in modes:
            if mode not in stats:
                continue
            row = dict(stats[mode])
            row["mode"] = mode
            row["failure_counts"] = json.dumps(row["failure_counts"], sort_keys=True)
            w.writerow({k: row.get(k, "") for k in fields})
    paths["csv"] = csv_path

    # summary.json
    json_path = os.path.join(out, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({m: stats[m] for m in modes if m in stats}, f, indent=2, sort_keys=True)
    paths["json"] = json_path

    # table.txt
    ascii_path = os.path.join(out, "table.txt")
    with open(ascii_path, "w", encoding="utf-8") as f:
        f.write(_ascii_table(stats, modes) + "\n\n")
        f.write(_extended_stats_block(stats, modes) + "\n\n")
        f.write(_failure_report(stats, modes) + "\n")
    paths["ascii"] = ascii_path

    # table.tex
    caption = (
        "Sandwich task: plan, chat, and dialog communication modes with open-loop execution. "
        "SR = simulator success rate; 95\\% CI via Wilson score interval; "
        "steps and LLM calls reported as mean $\\pm$ std."
    )
    tex_path = os.path.join(out, "table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_latex_table(stats, modes, caption, "tab:sandwich_open_loop") + "\n")
    paths["latex"] = tex_path

    # failures.json
    fail_path = os.path.join(out, "failures.json")
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(
            {m: stats[m]["failure_counts"] for m in modes if m in stats},
            f, indent=2, sort_keys=True,
        )
    paths["failures"] = fail_path

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment configuration
    parser.add_argument(
        "--modes", nargs="+",
        choices=["plan", "chat", "dialog"],
        default=["plan", "chat", "dialog"],
        help="Which LLM planner modes to evaluate.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Episodes per (mode × seed) combination.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Random seeds; one independent run per seed. Total episodes = episodes × len(seeds).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use fake env + prompters (no MuJoCo, no API key).",
    )

    # LLM settings
    parser.add_argument("--llm-source", default="gpt-4",
                        help="LLM backend: gpt-4 | gpt-4o | gemini-2.0-flash-exp | ...")
    parser.add_argument("--api-key-path", default="",
                        help="Path to a text file containing the API key.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-replans", type=int, default=3,
                        help="Retries inside the prompter when LLM gives a bad plan.")
    parser.add_argument("--max-calls-per-round", type=int, default=6,
                        help="Max per-agent LLM calls for dialog mode.")

    # Simulator settings
    parser.add_argument("--task-goal", default="Assemble the sandwich in the required order.")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-sim-steps", type=int, default=5000)
    parser.add_argument(
        "--uncertainty",
        choices=["none", "heuristic", "policy_metadata"],
        default="policy_metadata",
    )

    # Output
    parser.add_argument("--output-dir", default="results/sandwich_open_loop")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save per-episode RRT paths and mp4 renders.")
    parser.add_argument("--latex-caption", default="",
                        help="Override LaTeX table caption.")
    parser.add_argument("--no-print-latex", action="store_true",
                        help="Suppress LaTeX table from stdout (still saved to file).")

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    print("=" * 72)
    print("CRIE-BT Sandwich Evaluation")
    print("  modes        : {}".format(args.modes))
    print("  seeds        : {}".format(args.seeds))
    print("  episodes     : {}".format(args.episodes))
    print("  total runs   : {}".format(len(args.modes) * len(args.seeds) * args.episodes))
    print("  dry_run      : {}".format(args.dry_run))
    if not args.dry_run:
        print("  llm_source   : {}".format(args.llm_source))
    print("  output_dir   : {}".format(args.output_dir))
    print("=" * 72)

    rows = run_evaluation(args)
    stats = aggregate(rows, args.modes)
    paths = write_outputs(stats, args.modes, args)

    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    print(_ascii_table(stats, args.modes))
    print()
    print(_extended_stats_block(stats, args.modes))
    print(_failure_report(stats, args.modes))
    if not args.no_print_latex:
        print()
        print("--- LaTeX table (table.tex) ---")
        with open(paths["latex"]) as f:
            print(f.read())

    print()
    print("Output files:")
    for key, path in sorted(paths.items()):
        print("  {:10s} {}".format(key + ":", os.path.abspath(path)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
