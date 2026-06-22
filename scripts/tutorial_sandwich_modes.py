#!/usr/bin/env python
"""CRIE-BT Tutorial: plan + open_loop, chat + open_loop, dialog + open_loop.

This self-contained tutorial demonstrates the three LLM planner communication
modes on the sandwich task using fake components — no MuJoCo installation is
required.  Run it to understand how the modes differ, then use the CLI
commands printed at the end to run the same experiment with real simulators
and real LLMs.

Usage:
    python scripts/tutorial_sandwich_modes.py [--verbose]
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Any, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rocobench.crie_bt.controllers import build_controller
from rocobench.crie_bt.legacy_tasks import (
    LEGACY_ACTION_PLAN,
    LegacyPromptPlanner,
    LegacyTaskRRTExecutorAdapter,
)
from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus


# ---------------------------------------------------------------------------
# ANSI colour helpers (gracefully degraded when terminal has no colour support)
# ---------------------------------------------------------------------------

def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


COLOUR = _supports_colour()
_RESET = "\033[0m" if COLOUR else ""
_BOLD = "\033[1m" if COLOUR else ""
_CYAN = "\033[96m" if COLOUR else ""
_GREEN = "\033[92m" if COLOUR else ""
_YELLOW = "\033[93m" if COLOUR else ""
_MAGENTA = "\033[95m" if COLOUR else ""
_RED = "\033[91m" if COLOUR else ""
_DIM = "\033[2m" if COLOUR else ""


def h1(text: str) -> None:
    border = "=" * 72
    print("\n{}{}{}{}\n{}\n{}{}{}{}".format(_BOLD, _CYAN, border, _RESET, text, _BOLD, _CYAN, border, _RESET))


def h2(text: str) -> None:
    print("\n{}{}{} {}{}".format(_BOLD, _YELLOW, "▶", text, _RESET))


def info(text: str) -> None:
    for line in textwrap.wrap(text, width=70):
        print("  " + line)


def code(text: str) -> None:
    print("  {}{}{}".format(_DIM, text, _RESET))


def ok(text: str) -> None:
    print("  {}{}✓  {}{}".format(_BOLD, _GREEN, text, _RESET))


def fail(text: str) -> None:
    print("  {}{}✗  {}{}".format(_BOLD, _RED, text, _RESET))


def result_line(key: str, value: Any) -> None:
    print("  {:30s}{}{}{}".format(key + ":", _MAGENTA, value, _RESET))


# ---------------------------------------------------------------------------
# Sandwich task fake components
# ---------------------------------------------------------------------------

AGENTS = ["Chad", "Dave"]

SANDWICH_TASK_GOAL = "Assemble the sandwich in the required order."

SANDWICH_CONTEXT = (
    "2 robots, Chad and Dave, together make a [bacon_sandwich].\n"
    "Recipe order: bread_slice1 → bacon → cheese → tomato → bread_slice2\n"
    "bread_slice1 must be PUT on cutting_board first.\n"
    "Chad can only reach the right side; Dave can only reach the left side."
)

SANDWICH_ACTION_PROMPT = (
    "[Action Options]\n"
    "1) PICK <obj>   — only if gripper is empty, only the next recipe item\n"
    "2) PUT <obj1> <obj2>  — obj2 can be another food, cutting_board, or table\n"
    "3) WAIT\n"
    "Output format: EXECUTE\\nNAME <agent> ACTION <action>  (one line per agent)"
)

# A realistic multi-step plan for the sandwich task
SANDWICH_PLAN_STEPS = [
    "EXECUTE\nNAME Dave ACTION PICK bread_slice1\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PUT bread_slice1 cutting_board\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Chad ACTION PICK bacon\nNAME Dave ACTION WAIT",
    "EXECUTE\nNAME Chad ACTION PUT bacon bread_slice1\nNAME Dave ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PICK cheese\nNAME Chad ACTION WAIT",
    "EXECUTE\nNAME Dave ACTION PUT cheese bacon\nNAME Chad ACTION WAIT",
]


class FakeSandwichEnv:
    """Sandwich environment stub with state machine simulation."""

    def __init__(self):
        self.robot_name_map = {"ur5e_robotiq": "Chad", "panda": "Dave"}
        self._reset_state()

    def _reset_state(self):
        self._grippers = {"Chad": None, "Dave": None}
        self._stack = []
        self._step = 0

    def reset(self):
        self._reset_state()
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
        done = any(obj == "cheese" for obj, _ in self._stack)
        return (1.0 if done else 0.0), done

    def _obs(self):
        class _Obs:
            def __init__(self, grippers, stack, step):
                self.grippers = dict(grippers)
                self.stack = list(stack)
                self.step = step
        return _Obs(self._grippers, self._stack, self._step)


class _FakeParser:
    def parse(self, obs, response):
        return True, "", ["compiled_plan"]


class _SimulatingExecutor:
    """Executes actions against the fake env and tracks history."""

    def __init__(self, env: FakeSandwichEnv, verbose: bool = False):
        self.env = env
        self.verbose = verbose
        self.history: List[str] = []

    def execute(self, plan, obs, artifact_dir=None):
        response = getattr(plan, "parsed_proposal", "")
        if not response:
            for call in getattr(plan, "calls", []):
                raw = getattr(call, "raw_response", "") or ""
                if raw:
                    response = raw
                    break
        self.history.append(response)
        self.env._step += 1
        return SkillExecutionResult(
            success=True,
            status=SkillExecutionStatus.SUCCESS,
            reason="",
            num_sim_steps=30,
            reward=0.5,
            done=False,
            info={},
        )


# ---------------------------------------------------------------------------
# Fake LLM prompters that simulate each communication mode
# ---------------------------------------------------------------------------

class _FakeLLMPathPlan:
    def __init__(self, text: str):
        self.parsed_proposal = text


class PlanModePrompter:
    """Simulates SingleThreadPrompter with comm_mode='plan'.

    A single centralised LLM call reasons about the full task and emits one
    EXECUTE block.  History of past actions is passed but no agent-to-agent
    dialogue occurs.
    """

    def __init__(self, step_response: str, verbose: bool = False):
        self._response = step_response
        self.verbose = verbose
        self.calls = 0

    def prompt_one_round(self, obs, save_path: str = ""):
        self.calls += 1
        if self.verbose:
            print()
            code("  [plan mode LLM call #{}]".format(self.calls))
            code("  System: You are a central planner for two robots making a sandwich.")
            code("  User:   Given the current scene, produce one round of robot actions.")
            code("  LLM  →  {}".format(self._response.replace(chr(10), " | ")))
        plan = _FakeLLMPathPlan(self._response)
        return True, [plan], ["no prior feedback"], [self._response]


class ChatModePrompter:
    """Simulates SingleThreadPrompter with comm_mode='chat'.

    Like 'plan' but maintains a conversation history; each call appends the
    previous action and any environment feedback before re-querying the LLM.
    The CRIE-BT path is identical to plan mode — the difference is in how the
    LLM prompt is constructed inside SingleThreadPrompter.
    """

    def __init__(self, step_response: str, verbose: bool = False):
        self._response = step_response
        self.verbose = verbose
        self._history: List[str] = []
        self.calls = 0

    def prompt_one_round(self, obs, save_path: str = ""):
        self.calls += 1
        if self.verbose:
            print()
            code("  [chat mode LLM call #{}]".format(self.calls))
            if self._history:
                code("  History: {}".format(" → ".join(self._history[-2:])))
            code("  System: You are a planner with full action history.")
            code("  LLM  →  {}".format(self._response.replace(chr(10), " | ")))
        self._history.append(self._response)
        plan = _FakeLLMPathPlan(self._response)
        return True, [plan], ["no prior feedback"], [self._response]


class DialogModePrompter:
    """Simulates DialogPrompter.

    Each agent has its own LLM call and they exchange messages until reaching
    consensus.  The final agreed EXECUTE block is returned.  This uses more
    LLM tokens but allows each agent to reason from its own perspective.
    """

    def __init__(self, step_response: str, verbose: bool = False):
        self._response = step_response
        self.verbose = verbose
        self.calls = 0

    def prompt_one_round(self, obs, save_path: str = ""):
        self.calls += 1
        if self.verbose:
            print()
            code("  [dialog mode — round {} — multi-agent conversation]".format(self.calls))
            code("  Chad  →  'I can reach the right side. I'll pick bacon next.'")
            code("  Dave  →  'I'm holding bread_slice1. I'll put it on cutting_board.'")
            code("  Chad  →  'Agreed. Let me wait while you do that.'")
            code("  [consensus reached]  →  {}".format(self._response.replace(chr(10), " | ")))
        plan = _FakeLLMPathPlan(self._response)
        return True, [plan], ["agreed"], [self._response]


# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------

def _run_mode(mode_label: str, planner_mode: str, prompter_cls, step_response: str, verbose: bool) -> dict:
    env = FakeSandwichEnv()
    prompter = prompter_cls(step_response, verbose=verbose)
    planner = LegacyPromptPlanner("sandwich", prompter, planner_mode, AGENTS)
    sim_executor = _SimulatingExecutor(env, verbose=verbose)
    adapter = LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id="sandwich",
        parser=_FakeParser(),
        executor=sim_executor,
    )
    controller = build_controller("open_loop", planner, adapter, uncertainty_mode="none")
    log = controller.run_episode(env, SANDWICH_TASK_GOAL, max_steps=20)
    log["_sim_history"] = sim_executor.history
    log["_prompter_calls"] = prompter.calls
    return log


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_execute_block(response: str) -> None:
    for line in response.strip().splitlines():
        code("    " + line)


def _print_log(log: dict, step_response: str, verbose: bool) -> None:
    print()
    result_line("  controller mode", log["mode"])
    result_line("  planner calls", log["planner_calls"])
    result_line("  LLM prompter calls", log["_prompter_calls"])
    result_line("  steps executed", log["steps"])
    result_line("  completed subtasks", log["completed_subtasks"])
    result_line("  failed subtasks", log["failed_subtasks"])

    if log.get("subtask_results"):
        r = log["subtask_results"][0]
        result_line("  subtask status", r["status"])
        result_line("  uncertainty (confidence)", r["uncertainty"]["confidence"])
        result_line("  uncertainty (risk_level)", r["uncertainty"]["risk_level"])

    print()
    if log["success"]:
        ok("Episode succeeded")
    else:
        fail("Episode failed: {}".format(log.get("explanations", ["-"])[0] if log.get("explanations") else "see failure_counts"))

    if verbose:
        print()
        info("EXECUTE block sent to the simulator:")
        _print_execute_block(step_response)

    if log.get("failure_counts"):
        result_line("  failure_counts", log["failure_counts"])


# ---------------------------------------------------------------------------
# CLI reference
# ---------------------------------------------------------------------------

CLI_REFERENCE = """
{bold}{cyan}───  CLI COMMANDS (real MuJoCo + LLM)  ──────────────────────────────────{reset}

{dim}# plan + open_loop — central single-shot planner{reset}
python scripts/run_crie_bt_sim.py \\
    --task sandwich \\
    --planner-mode plan \\
    --mode open_loop \\
    --llm-source gemini-2.0-flash-exp \\
    --num-replans 3 \\
    --output artifacts/crie_bt/sandwich_plan_open_loop.jsonl \\
    --artifact-dir artifacts/crie_bt/sandwich_plan_open_loop_$(date +%Y%m%d_%H%M%S)

{dim}# chat + open_loop — planner with action history{reset}
python scripts/run_crie_bt_sim.py \\
    --task sandwich \\
    --planner-mode chat \\
    --mode open_loop \\
    --llm-source gemini-2.0-flash-exp \\
    --num-replans 3 \\
    --output artifacts/crie_bt/sandwich_chat_open_loop.jsonl \\
    --artifact-dir artifacts/crie_bt/sandwich_chat_open_loop_$(date +%Y%m%d_%H%M%S)

{dim}# dialog + open_loop — per-agent dialogue then consensus{reset}
python scripts/run_crie_bt_sim.py \\
    --task sandwich \\
    --planner-mode dialog \\
    --mode open_loop \\
    --llm-source gemini-2.0-flash-exp \\
    --max-calls-per-round 6 \\
    --num-replans 3 \\
    --output artifacts/crie_bt/sandwich_dialog_open_loop.jsonl \\
    --artifact-dir artifacts/crie_bt/sandwich_dialog_open_loop_$(date +%Y%m%d_%H%M%S)

{bold}{cyan}Relevant flags:{reset}
  --llm-source   gemini-2.0-flash-exp | gpt-4 | gpt-4o | ...
  --api-key-path path to a text file containing the API key
  --num-replans  how many times the prompter retries if LLM gives a bad plan
  --temperature  LLM temperature (default 0.0 for reproducibility)
  --episodes     number of independent simulation episodes to run
  --seed         random seed for env initialisation
  --artifact-dir directory where prompt logs / mp4s are saved
""".format(
    bold=_BOLD, cyan=_CYAN, reset=_RESET, dim=_DIM
)


# ---------------------------------------------------------------------------
# Tutorial scenarios
# ---------------------------------------------------------------------------

TUTORIAL_MODES = [
    {
        "label": "plan + open_loop",
        "planner_mode": "plan",
        "prompter_cls": PlanModePrompter,
        "step_response": SANDWICH_PLAN_STEPS[0],
        "concept": (
            "A single centralised LLM call reasons about the full task state and produces "
            "one EXECUTE block per round.  No agent-to-agent dialogue occurs.  This is the "
            "fastest and cheapest mode — it uses the fewest tokens per round but agents "
            "cannot express individual perspectives."
        ),
        "when_to_use": (
            "Use 'plan' when task structure is well-defined, actions are easily verifiable, "
            "and minimising LLM cost per step matters more than agent autonomy."
        ),
    },
    {
        "label": "chat + open_loop",
        "planner_mode": "chat",
        "prompter_cls": ChatModePrompter,
        "step_response": SANDWICH_PLAN_STEPS[1],
        "concept": (
            "Like 'plan', but the LLM prompt includes a growing history of prior actions and "
            "environment feedback.  Each call builds on what happened before, enabling the "
            "planner to reason about progress and adapt without full replanning.  Still a "
            "single LLM call per round."
        ),
        "when_to_use": (
            "Use 'chat' when task history matters (e.g. knowing whether bacon was already "
            "placed influences what to do next) but agent-to-agent negotiation isn't needed."
        ),
    },
    {
        "label": "dialog + open_loop",
        "planner_mode": "dialog",
        "prompter_cls": DialogModePrompter,
        "step_response": SANDWICH_PLAN_STEPS[2],
        "concept": (
            "DialogPrompter gives each agent its own LLM call and lets them exchange messages "
            "until reaching consensus.  This is the most expressive mode — each agent can "
            "reason from its own reachability constraints (Chad: right side, Dave: left side) "
            "and propose actions.  Consensus is reached after up to max_calls_per_round turns."
        ),
        "when_to_use": (
            "Use 'dialog' when agent heterogeneity matters — different reach, capabilities, "
            "or roles — and you can afford the extra LLM calls.  Produces richer coordination "
            "traces useful for HRC studies."
        ),
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show simulated LLM dialogue and EXECUTE blocks")
    args = parser.parse_args(argv)

    h1("CRIE-BT Tutorial  —  Sandwich Task")

    info("The CRIE-BT framework wraps RoCoBench environments and LLM prompters")
    info("inside a Behavior Tree controller.  This tutorial shows all three LLM")
    info("communication modes paired with the open_loop execution controller.")
    print()
    info("Sandwich task:  Chad + Dave assemble a bacon sandwich in recipe order.")
    info("Recipe:  bread_slice1 → bacon → cheese → tomato → bread_slice2")
    info("Constraint:  Chad reaches right side only; Dave reaches left side only.")

    results = {}

    for scenario in TUTORIAL_MODES:
        h2("{} — {}".format(scenario["label"].upper(), scenario["prompter_cls"].__doc__.strip().splitlines()[0]))

        info(scenario["concept"])
        print()
        info("When to use: " + scenario["when_to_use"])

        log = _run_mode(
            scenario["label"],
            scenario["planner_mode"],
            scenario["prompter_cls"],
            scenario["step_response"],
            verbose=args.verbose,
        )
        _print_log(log, scenario["step_response"], verbose=args.verbose)
        results[scenario["label"]] = log

    # Summary comparison table
    h1("Mode Comparison Summary")
    print("{}{:<25s}  {:>12s}  {:>14s}  {:>10s}  {:>12s}{}".format(
        _BOLD,
        "Mode", "LLM calls/step", "Steps executed", "Success", "Confidence",
        _RESET,
    ))
    print("-" * 78)
    for label, log in results.items():
        confidence = "—"
        if log.get("subtask_results"):
            confidence = "{:.2f}".format(log["subtask_results"][0]["uncertainty"]["confidence"])
        print("{:<25s}  {:>12d}  {:>14d}  {:>10s}  {:>12s}".format(
            label,
            log["_prompter_calls"],
            log["steps"],
            "✓" if log["success"] else "✗",
            confidence,
        ))

    # Key differences explanation
    h2("Key architectural differences")

    rows = [
        ("Planner type",      "SingleThreadPrompter",   "SingleThreadPrompter",  "DialogPrompter"),
        ("LLM calls/round",   "1",                      "1",                     "1 per agent"),
        ("History passed",    "No",                     "Yes",                   "Yes (per agent)"),
        ("Agent dialogue",    "No",                     "No",                    "Yes"),
        ("Token cost",        "Lowest",                 "Medium",                "Highest"),
        ("Agent autonomy",    "None",                   "None",                  "Per-agent perspective"),
        ("CRIE-BT path",      "Identical",              "Identical",             "Identical"),
    ]
    header = "{}{:<25s}  {:<22s}  {:<22s}  {:<22s}{}".format(_BOLD, "Property", "plan", "chat", "dialog", _RESET)
    print(header)
    print("-" * 95)
    for row in rows:
        print("{:<25s}  {:<22s}  {:<22s}  {:<22s}".format(*row))

    print()
    info("Note: all three modes produce the same CRIE-BT log schema and are")
    info("executed by the same OpenLoopController.  The only difference is HOW")
    info("the LLM prompt is constructed inside the planner (LegacyPromptPlanner).")

    print(CLI_REFERENCE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
