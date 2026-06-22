"""Tests for plan/chat/dialog + open_loop on the sandwich task.

All three planner modes (plan, chat, dialog) are exercised with the
OpenLoopController using fully self-contained fakes — no MuJoCo required.
The sandwich scenario matches the real MakeSandwichTask:
  - Agents: Chad (right side), Dave (left side)
  - Recipe: bread_slice1 → bacon → cheese → tomato → bread_slice2
  - bread_slice1 must be PUT on cutting_board first
"""

import copy
from typing import Any, List, Optional, Tuple

from rocobench.crie_bt.controllers import build_controller
from rocobench.crie_bt.legacy_tasks import (
    LEGACY_ACTION_PLAN,
    LegacyActionPlanner,
    LegacyPromptPlanner,
    LegacyTaskRRTExecutorAdapter,
    legacy_response_from_prompt_outputs,
    make_wait_response,
    split_legacy_responses,
)
from rocobench.crie_bt.status import BTStatus, FailureCode
from rocobench.crie_bt.types import ExecutionContext, SkillCall
from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus


# ---------------------------------------------------------------------------
# Sandwich task fake environment
# ---------------------------------------------------------------------------

SANDWICH_AGENTS = ["Chad", "Dave"]
SANDWICH_TASK_CONTEXT = (
    "2 robots, Chad and Dave, together make a [bacon_sandwich].\n"
    "Food items must be stacked following this order: bread_slice1, bacon, cheese, tomato, bread_slice2, "
    "where bread_slice1 must be PUT on cutting_board.\n"
    "Chad can only reach right side of the table, and Dave can only reach left side of the table.\n"
    "Both robots can PICK food items, or PUT an item atop something; only one robot can PUT at a time.\n"
    "At each round, given [Scene description] and [Environment feedback], use it to reason about the task and improve plans."
)
SANDWICH_ACTION_PROMPT = (
    "\n[Action Options]\n"
    "1) PICK <obj>, Only PICK if gripper is empty. PICK only the correct next item according to the recipe.\n"
    "2) PUT <obj1> <obj2>. <obj1> can be one of the foods. <obj2> can be food, cutting_board, or table.\n"
    "3) WAIT, do nothing.\n"
    "Only one robot can PUT each round. You must PICK up an item before PUT.\n"
    "[Action Output Instruction]\n"
    "Must first output 'EXECUTE\\n', then give exactly one action per robot, put each on a new line.\n"
)


class FakeSandwichEnv:
    """Minimal sandwich environment stub with stateful gripper tracking."""

    def __init__(self):
        self.robot_name_map = {"ur5e_robotiq": "Chad", "panda": "Dave"}
        self._done = False
        self._step = 0
        self._grippers = {"Chad": None, "Dave": None}
        self._stack = []

    # --- RoCoBench env interface ---

    def reset(self):
        self._done = False
        self._step = 0
        self._grippers = {"Chad": None, "Dave": None}
        self._stack = []
        return self._make_obs()

    def get_obs(self):
        return self._make_obs()

    def get_action_prompt(self):
        return SANDWICH_ACTION_PROMPT

    def describe_task_context(self):
        return SANDWICH_TASK_CONTEXT

    def describe_obs(self, obs):
        lines = ["[Scene description]"]
        for item, agent in self._grippers.items():
            if agent:
                lines.append("{}'s gripper is holding {}".format(item, agent))
            else:
                lines.append("{}'s gripper is empty".format(item))
        lines.append("Stack: {}".format(self._stack))
        return "\n".join(lines)

    def get_reward_done(self, obs):
        del obs
        return (1.0 if self._done else 0.0), self._done

    def get_task_feedback(self, path_plan, pose_dict=None):
        return ""

    def seed(self, np_seed=0):
        pass

    # --- Simulate an action ---

    def apply_action(self, agent: str, action: str) -> bool:
        """Return True if the action was valid and applied."""
        action = action.strip()
        if action == "WAIT":
            return True
        if action.startswith("PICK "):
            obj = action[5:].strip()
            if self._grippers[agent] is not None:
                return False
            self._grippers[agent] = obj
            return True
        if action.startswith("PUT "):
            parts = action[4:].strip().split()
            if len(parts) != 2:
                return False
            obj, target = parts
            if self._grippers[agent] != obj:
                return False
            self._grippers[agent] = None
            self._stack.append((obj, target))
            # Mark done when bread_slice2 placed
            if obj == "bread_slice2":
                self._done = True
            return True
        return False

    def _make_obs(self):
        obs = _FakeObs(self._grippers.copy(), list(self._stack), self._done)
        return obs


class _FakeObs:
    def __init__(self, grippers, stack, done):
        self.grippers = grippers
        self.stack = stack
        self.done = done


# ---------------------------------------------------------------------------
# Fake executor / parser
# ---------------------------------------------------------------------------

class _FakeParser:
    """Always succeeds; passes the response through unchanged."""

    def __init__(self, success: bool = True, reason: str = ""):
        self.success = success
        self.reason = reason

    def parse(self, obs, response):
        if not self.success:
            return False, self.reason, []
        return True, "", ["compiled_plan"]


class _ActionTrackingExecutor:
    """Applies actions to the FakeSandwichEnv and returns appropriate results."""

    def __init__(self, env: FakeSandwichEnv, success: bool = True):
        self.env = env
        self.success = success
        self.executed = []  # history of responses executed

    def execute(self, plan, obs, artifact_dir=None):
        del obs, artifact_dir
        self.executed.append(plan)
        status = SkillExecutionStatus.SUCCESS if self.success else SkillExecutionStatus.MOTION_PLANNING_FAILED
        return SkillExecutionResult(
            success=self.success,
            status=status,
            reason="" if self.success else "motion planning failed",
            num_sim_steps=30,
            reward=1.0 if self.success else 0.0,
            done=self.env._done,
            info={"is_success": self.success},
        )


# ---------------------------------------------------------------------------
# Fake LLM prompters
# ---------------------------------------------------------------------------

class _FakeLLMPathPlan:
    def __init__(self, parsed_proposal: str):
        self.parsed_proposal = parsed_proposal


class _SingleStepPrompter:
    """Returns a fixed EXECUTE block on every call (simulates plan/chat mode)."""

    def __init__(self, response: str):
        self.response = response
        self.calls: List[Tuple[Any, str]] = []

    def prompt_one_round(self, obs, save_path: str = ""):
        self.calls.append((obs, save_path))
        plan = _FakeLLMPathPlan(self.response)
        return True, [plan], ["no feedback"], [self.response]


class _SequentialPrompter:
    """Returns successive EXECUTE blocks from a list (simulates multi-step chat)."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[Any] = []
        self._idx = 0

    def prompt_one_round(self, obs, save_path: str = ""):
        self.calls.append((obs, save_path))
        idx = min(self._idx, len(self.responses) - 1)
        response = self.responses[idx]
        self._idx += 1
        plan = _FakeLLMPathPlan(response)
        return True, [plan], ["no feedback"], [response]


class _FailingPrompter:
    """Simulates an LLM that fails to produce a plan."""

    def prompt_one_round(self, obs, save_path: str = ""):
        return False, [], [], []


class _NoExecutePrompter:
    """Returns output with no EXECUTE block."""

    def prompt_one_round(self, obs, save_path: str = ""):
        raw = "I am thinking about the sandwich task..."
        plan = _FakeLLMPathPlan(raw)
        return True, [plan], [], [raw]


# ---------------------------------------------------------------------------
# Helper to build adapter+controller
# ---------------------------------------------------------------------------

def _build_combo(env, planner, executor_obj=None, mode="open_loop"):
    if executor_obj is None:
        executor_obj = _ActionTrackingExecutor(env)
    adapter = LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id="sandwich",
        parser=_FakeParser(),
        executor=executor_obj,
    )
    return build_controller(mode, planner, adapter, uncertainty_mode="none")


# ---------------------------------------------------------------------------
# Sandwich-specific EXECUTE blocks
# ---------------------------------------------------------------------------

STEP1_PICK = "EXECUTE\nNAME Dave ACTION PICK bread_slice1\nNAME Chad ACTION WAIT"
STEP2_PUT = "EXECUTE\nNAME Dave ACTION PUT bread_slice1 cutting_board\nNAME Chad ACTION WAIT"
STEP3_PICK_BACON = "EXECUTE\nNAME Chad ACTION PICK bacon\nNAME Dave ACTION WAIT"
STEP4_PUT_BACON = "EXECUTE\nNAME Chad ACTION PUT bacon bread_slice1\nNAME Dave ACTION WAIT"


# ===========================================================================
# plan + open_loop
# ===========================================================================

class TestPlanOpenLoop:
    def test_plan_mode_generates_single_plan_and_executes(self):
        env = FakeSandwichEnv()
        prompter = _SingleStepPrompter(STEP1_PICK)
        planner = LegacyPromptPlanner("sandwich", prompter, "plan", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "Assemble the sandwich in the required order.", max_steps=5)

        assert row["mode"] == "open_loop"
        assert row["planner_calls"] == 1
        assert row["replans"] == 0
        assert row["completed_subtasks"] == 1
        assert row["failed_subtasks"] == 0
        assert row["success"]
        assert len(prompter.calls) == 1

    def test_plan_mode_saves_artifacts_under_call_directory(self, tmp_path):
        env = FakeSandwichEnv()
        prompter = _SingleStepPrompter(STEP1_PICK)
        planner = LegacyPromptPlanner("sandwich", prompter, "plan", SANDWICH_AGENTS, save_dir=str(tmp_path))
        controller = _build_combo(env, planner)

        controller.run_episode(env, "Assemble the sandwich.", max_steps=5)

        assert (tmp_path / "planner_call_001").is_dir()

    def test_plan_mode_records_skill_call_in_subtask_results(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), "plan", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        result = row["subtask_results"][0]
        assert result["skill_call"]["skill_name"] == LEGACY_ACTION_PLAN
        assert result["status"] == "SUCCESS"
        assert "bread_slice1" in result["skill_call"]["arguments"]["response"]

    def test_plan_mode_propagates_executor_failure(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), "plan", SANDWICH_AGENTS)
        broken_exec = _ActionTrackingExecutor(env, success=False)
        controller = _build_combo(env, planner, executor_obj=broken_exec)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        assert not row["success"]
        assert row["failed_subtasks"] == 1
        assert row["completed_subtasks"] == 0

    def test_plan_mode_handles_prompter_failure_gracefully(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _FailingPrompter(), "plan", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        assert not row["success"]
        assert "PLANNER_ERROR" in row["failure_counts"]
        assert row["planner_calls"] == 1

    def test_plan_mode_parser_rejection_is_a_step_failure_not_planner_error(self):
        """If LLM output has no valid EXECUTE block, the parser rejects it as POSTCONDITION_FAILED."""
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), "plan", SANDWICH_AGENTS)
        bad_parser = _FakeParser(success=False, reason="missing EXECUTE keyword")
        adapter = LegacyTaskRRTExecutorAdapter(
            env=env, task_id="sandwich", parser=bad_parser,
            executor=_ActionTrackingExecutor(env),
        )
        controller = build_controller("open_loop", planner, adapter, uncertainty_mode="none")

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        assert not row["success"]
        assert "PLANNER_ERROR" not in row["failure_counts"]
        assert "POSTCONDITION_FAILED" in row["failure_counts"]


# ===========================================================================
# chat + open_loop
# ===========================================================================

class TestChatOpenLoop:
    def test_chat_mode_generates_single_plan_and_executes(self):
        env = FakeSandwichEnv()
        prompter = _SingleStepPrompter(STEP2_PUT)
        planner = LegacyPromptPlanner("sandwich", prompter, "chat", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "Assemble the sandwich.", max_steps=5)

        assert row["mode"] == "open_loop"
        assert row["planner_calls"] == 1
        assert row["success"]
        assert len(prompter.calls) == 1

    def test_chat_mode_uses_same_execute_path_as_plan_mode(self):
        """chat and plan differ in comm_mode only — the CRIE-BT path is identical."""
        env_plan = FakeSandwichEnv()
        env_chat = FakeSandwichEnv()
        response = STEP3_PICK_BACON

        planner_plan = LegacyPromptPlanner("sandwich", _SingleStepPrompter(response), "plan", SANDWICH_AGENTS)
        planner_chat = LegacyPromptPlanner("sandwich", _SingleStepPrompter(response), "chat", SANDWICH_AGENTS)

        row_plan = _build_combo(env_plan, planner_plan).run_episode(env_plan, "sandwich", max_steps=5)
        row_chat = _build_combo(env_chat, planner_chat).run_episode(env_chat, "sandwich", max_steps=5)

        # Both modes produce identical CRIE-BT log structure
        assert row_plan["success"] == row_chat["success"]
        assert row_plan["planner_calls"] == row_chat["planner_calls"]
        assert row_plan["mode"] == row_chat["mode"]

    def test_chat_mode_passes_observation_to_prompter(self):
        env = FakeSandwichEnv()
        prompter = _SingleStepPrompter(STEP1_PICK)
        planner = LegacyPromptPlanner("sandwich", prompter, "chat", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        obs_before = env.get_obs()
        controller.run_episode(env, "sandwich task", max_steps=5)

        called_obs, _ = prompter.calls[0]
        # Prompter receives an observation object, not None
        assert called_obs is not None


# ===========================================================================
# dialog + open_loop
# ===========================================================================

class TestDialogOpenLoop:
    def test_dialog_mode_generates_single_plan_and_executes(self):
        env = FakeSandwichEnv()
        prompter = _SingleStepPrompter(STEP4_PUT_BACON)
        planner = LegacyPromptPlanner("sandwich", prompter, "dialog", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "Assemble the sandwich.", max_steps=5)

        assert row["mode"] == "open_loop"
        assert row["planner_calls"] == 1
        assert row["success"]
        assert row["completed_subtasks"] == 1

    def test_dialog_mode_metadata_records_planner_type(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), "dialog", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        result = row["subtask_results"][0]
        # The step_id includes the planner mode name
        assert "dialog" in result["skill_call"]["instruction"]

    def test_dialog_mode_handles_prompter_failure_gracefully(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _FailingPrompter(), "dialog", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        assert not row["success"]
        assert "PLANNER_ERROR" in row["failure_counts"]


# ===========================================================================
# Cross-mode comparison
# ===========================================================================

class TestAllThreeModes:
    def test_all_three_modes_produce_same_log_schema(self):
        required_keys = {
            "mode", "task", "success", "steps", "planner_calls", "replans",
            "completed_subtasks", "failed_subtasks", "failure_counts",
            "events", "subtask_results", "explanations",
        }
        for planner_mode in ("plan", "chat", "dialog"):
            env = FakeSandwichEnv()
            planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), planner_mode, SANDWICH_AGENTS)
            row = _build_combo(env, planner).run_episode(env, "sandwich task", max_steps=5)
            missing = required_keys - set(row.keys())
            assert not missing, "Mode {} missing keys: {}".format(planner_mode, missing)

    def test_all_three_modes_use_open_loop_execution(self):
        for planner_mode in ("plan", "chat", "dialog"):
            env = FakeSandwichEnv()
            planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), planner_mode, SANDWICH_AGENTS)
            row = _build_combo(env, planner).run_episode(env, "sandwich task", max_steps=5)
            assert row["mode"] == "open_loop", "Expected open_loop for mode={}".format(planner_mode)
            assert row["replans"] == 0, "Open loop should never replan"

    def test_all_three_modes_call_prompter_exactly_once(self):
        for planner_mode in ("plan", "chat", "dialog"):
            env = FakeSandwichEnv()
            prompter = _SingleStepPrompter(STEP1_PICK)
            planner = LegacyPromptPlanner("sandwich", prompter, planner_mode, SANDWICH_AGENTS)
            _build_combo(env, planner).run_episode(env, "sandwich task", max_steps=5)
            assert len(prompter.calls) == 1, "Mode {} should call prompter once".format(planner_mode)

    def test_all_three_modes_fail_gracefully_when_planner_errors(self):
        for planner_mode in ("plan", "chat", "dialog"):
            env = FakeSandwichEnv()
            planner = LegacyPromptPlanner("sandwich", _FailingPrompter(), planner_mode, SANDWICH_AGENTS)
            row = _build_combo(env, planner).run_episode(env, "sandwich task", max_steps=5)
            assert not row["success"], "Planner error should result in failure for mode={}".format(planner_mode)
            assert "PLANNER_ERROR" in row["failure_counts"], "Should record planner error for mode={}".format(planner_mode)


# ===========================================================================
# Multi-step sandwich sequence
# ===========================================================================

class TestMultiStepSandwichSequence:
    def test_action_planner_executes_full_two_step_sequence(self):
        """LegacyActionPlanner with two scripted EXECUTE blocks runs both steps."""
        env = FakeSandwichEnv()
        responses = [STEP1_PICK, STEP2_PUT]
        planner = LegacyActionPlanner("sandwich", responses=responses, agent_names=SANDWICH_AGENTS)
        executor_obj = _ActionTrackingExecutor(env)
        controller = _build_combo(env, planner, executor_obj=executor_obj)

        row = controller.run_episode(env, "Assemble the sandwich.", max_steps=10)

        assert row["planner_calls"] == 1
        assert row["completed_subtasks"] == 2
        assert row["success"]
        assert len(executor_obj.executed) == 2

    def test_action_planner_stops_at_first_failure(self):
        """When step 1 fails, open_loop stops and does not proceed to step 2."""
        env = FakeSandwichEnv()
        responses = [STEP1_PICK, STEP2_PUT]
        planner = LegacyActionPlanner("sandwich", responses=responses, agent_names=SANDWICH_AGENTS)
        broken_exec = _ActionTrackingExecutor(env, success=False)
        controller = _build_combo(env, planner, executor_obj=broken_exec)

        row = controller.run_episode(env, "Assemble the sandwich.", max_steps=10)

        assert row["failed_subtasks"] == 1
        assert row["completed_subtasks"] == 0
        assert not row["success"]
        assert len(broken_exec.executed) == 1

    def test_sequential_prompter_is_called_once_per_generate_plan(self):
        """LegacyPromptPlanner wraps a single prompter call into one CollaborativePlan step."""
        env = FakeSandwichEnv()
        prompter = _SequentialPrompter([STEP1_PICK, STEP2_PUT])
        planner = LegacyPromptPlanner("sandwich", prompter, "plan", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        # open_loop calls generate_plan once
        row = controller.run_episode(env, "sandwich task", max_steps=5)

        assert len(prompter.calls) == 1
        assert row["planner_calls"] == 1
        assert row["completed_subtasks"] == 1


# ===========================================================================
# Uncertainty metadata
# ===========================================================================

class TestUncertaintyMetadata:
    def test_success_reports_high_confidence_uncertainty(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), "plan", SANDWICH_AGENTS)
        controller = _build_combo(env, planner)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        result = row["subtask_results"][0]
        assert result["uncertainty"]["confidence"] > 0.5
        assert result["uncertainty"]["risk_level"] == "low"
        assert result["raw_info"]["fake_uncertainty"]

    def test_failure_reports_low_confidence_uncertainty(self):
        env = FakeSandwichEnv()
        planner = LegacyPromptPlanner("sandwich", _SingleStepPrompter(STEP1_PICK), "plan", SANDWICH_AGENTS)
        broken = _ActionTrackingExecutor(env, success=False)
        controller = _build_combo(env, planner, executor_obj=broken)

        row = controller.run_episode(env, "sandwich task", max_steps=5)

        result = row["subtask_results"][0]
        assert result["uncertainty"]["confidence"] < 0.5
        assert result["uncertainty"]["risk_level"] in ("medium", "high")


# ===========================================================================
# Task spec and adapter metadata
# ===========================================================================

class TestSandwichTaskSpec:
    def test_legacy_task_spec_contains_sandwich_skills_and_context(self):
        from rocobench.crie_bt.legacy_tasks import legacy_task_spec
        env = FakeSandwichEnv()
        spec = legacy_task_spec(env, "sandwich")

        assert spec["task"] == "sandwich"
        assert spec["adapter"] == "legacy_action_plan"
        assert LEGACY_ACTION_PLAN in spec["skills"]
        assert spec["agents"] == SANDWICH_AGENTS
        assert "PICK" in spec["actions"]["prompt"]
        assert "PUT" in spec["actions"]["prompt"]
