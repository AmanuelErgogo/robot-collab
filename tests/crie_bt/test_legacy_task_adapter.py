from rocobench.crie_bt.legacy_tasks import (
    LEGACY_ACTION_PLAN,
    LegacyActionPlanner,
    LegacyPromptPlanner,
    LegacyTaskRRTExecutorAdapter,
    available_legacy_task_ids,
    legacy_response_from_prompt_outputs,
    legacy_task_spec,
    make_wait_response,
    normalize_task_id,
    split_legacy_responses,
)
from rocobench.crie_bt.controllers import build_controller
from rocobench.crie_bt.status import BTStatus, FailureCode
from rocobench.crie_bt.types import ExecutionContext, SkillCall
from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus


class FakeLegacyEnv:
    def __init__(self):
        self.robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}
        self.obs = object()
        self.done = False

    def get_action_prompt(self):
        return "NAME Alice ACTION WAIT\nNAME Bob ACTION WAIT"

    def describe_task_context(self):
        return "Fake task context."

    def get_obs(self):
        return self.obs

    def get_reward_done(self, obs):
        del obs
        return 0.0, self.done

    def get_task_feedback(self, llm_plan, pose_dict):
        del llm_plan, pose_dict
        return ""


class BrokenMetadataEnv(FakeLegacyEnv):
    def describe_task_context(self):
        raise AttributeError("missing recipe_name")


class FakeParser:
    def __init__(self, success=True, reason=""):
        self.success = success
        self.reason = reason

    def parse(self, obs, response):
        del obs, response
        if not self.success:
            return False, self.reason, []
        return True, "", ["compiled_path_plan"]


class FakeExecutor:
    def __init__(self, env, success=True, status=SkillExecutionStatus.SUCCESS):
        self.env = env
        self.success = success
        self.status = status

    def execute(self, plan, obs, artifact_dir=None):
        del plan, obs, artifact_dir
        return SkillExecutionResult(
            success=self.success,
            status=self.status,
            reason="" if self.success else "execution failed",
            num_sim_steps=2,
            reward=1.0 if self.success else 0.0,
            done=self.env.done,
            info={"is_success": self.success},
        )


class FakeLLMPathPlan:
    def __init__(self, parsed_proposal):
        self.parsed_proposal = parsed_proposal


class FakePrompter:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def prompt_one_round(self, obs, save_path=""):
        self.calls.append((obs, save_path))
        plan = FakeLLMPathPlan("NAME Alice ACTION WAIT\nNAME Bob ACTION WAIT")
        return True, [plan], ["None"], [self.response]


def test_legacy_task_ids_cover_existing_rocobench_tasks():
    assert set(available_legacy_task_ids()) == {"pack", "sort", "sweep", "sandwich", "rope", "cabinet"}
    assert normalize_task_id("pack_grocery") == "pack"
    assert normalize_task_id("make_sandwich") == "sandwich"


def test_make_wait_response_emits_one_action_per_agent():
    response = make_wait_response(["Alice", "Bob"])

    assert response.startswith("EXECUTE")
    assert "NAME Alice ACTION WAIT" in response
    assert "NAME Bob ACTION WAIT" in response


def test_split_legacy_responses_accepts_multiple_execute_blocks():
    text = """EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION PICK cube
EXECUTE
NAME Alice ACTION PUT cube bin
NAME Bob ACTION WAIT
"""

    responses = split_legacy_responses(text)

    assert len(responses) == 2
    assert responses[0].startswith("EXECUTE")
    assert "PICK cube" in responses[0]
    assert "PUT cube bin" in responses[1]


def test_legacy_response_from_prompt_outputs_prefers_raw_execute_block():
    plan = FakeLLMPathPlan("NAME Alice ACTION WAIT\nNAME Bob ACTION WAIT")
    response = legacy_response_from_prompt_outputs([plan], ["reasoning\nEXECUTE\nNAME Alice ACTION PICK cube\nNAME Bob ACTION WAIT"])

    assert response == "EXECUTE\nNAME Alice ACTION PICK cube\nNAME Bob ACTION WAIT"


def test_legacy_response_from_prompt_outputs_falls_back_to_parsed_proposal():
    plan = FakeLLMPathPlan("NAME Alice ACTION WAIT\nNAME Bob ACTION WAIT")
    response = legacy_response_from_prompt_outputs([plan], [])

    assert response == "EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT"


def test_legacy_task_spec_exposes_action_plan_contract():
    spec = legacy_task_spec(FakeLegacyEnv(), "sort")

    assert spec["task"] == "sort"
    assert spec["adapter"] == "legacy_action_plan"
    assert LEGACY_ACTION_PLAN in spec["skills"]
    assert spec["skills"][LEGACY_ACTION_PLAN]["subtask_success"] == "legacy parser succeeds and RRT executor succeeds"


def test_legacy_task_spec_tolerates_broken_optional_metadata():
    spec = legacy_task_spec(BrokenMetadataEnv(), "sandwich")

    assert spec["task"] == "sandwich"
    assert "describe_task_context unavailable" in spec["task_context"]


def test_legacy_action_planner_wraps_raw_response_as_one_skill_call():
    response = make_wait_response(["Alice", "Bob"])
    planner = LegacyActionPlanner("sort", response=response, agent_names=["Alice", "Bob"])

    plan = planner.generate_plan("sort block", object())

    assert len(plan.steps) == 1
    assert plan.steps[0].skill_call.skill_name == LEGACY_ACTION_PLAN
    assert plan.steps[0].skill_call.arguments["response"] == response
    assert plan.steps[0].role_assignment["Alice"] == "legacy_action_agent"


def test_legacy_action_planner_wraps_multiple_responses_as_steps():
    responses = [
        "EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION PICK cube",
        "EXECUTE\nNAME Alice ACTION PUT cube bin\nNAME Bob ACTION WAIT",
    ]
    planner = LegacyActionPlanner("sort", responses=responses, agent_names=["Alice", "Bob"])

    plan = planner.generate_plan("sort block", object())

    assert len(plan.steps) == 2
    assert plan.steps[0].skill_call.arguments["response"] == responses[0]
    assert plan.steps[1].skill_call.arguments["response"] == responses[1]
    assert plan.metadata["responses"] == responses


def test_legacy_prompt_planner_wraps_plan_mode_output(tmp_path):
    prompter = FakePrompter("central reasoning\nEXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT")
    planner = LegacyPromptPlanner("sort", prompter, "plan", ["Alice", "Bob"], save_dir=str(tmp_path))

    plan = planner.generate_plan("sort block", object())

    assert len(plan.steps) == 1
    assert plan.steps[0].skill_call.skill_name == LEGACY_ACTION_PLAN
    assert plan.steps[0].skill_call.arguments["response"].startswith("EXECUTE")
    assert plan.metadata["planner"] == "plan"
    assert prompter.calls[0][1].endswith("planner_call_001")


def test_legacy_prompt_planner_wraps_chat_and_dialog_modes():
    for mode in ("chat", "dialog"):
        prompter = FakePrompter("[Agent]:\nEXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT")
        planner = LegacyPromptPlanner("sort", prompter, mode, ["Alice", "Bob"])

        plan = planner.generate_plan("sort block", object())

        assert len(plan.steps) == 1
        assert plan.metadata["planner"] == mode
        assert "NAME Bob ACTION WAIT" in plan.steps[0].skill_call.arguments["response"]


def test_plan_chat_dialog_prompt_planners_run_with_open_loop_controller():
    for mode in ("plan", "chat", "dialog"):
        env = FakeLegacyEnv()
        prompter = FakePrompter("reasoning\nEXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT")
        planner = LegacyPromptPlanner("sort", prompter, mode, ["Alice", "Bob"])
        executor = LegacyTaskRRTExecutorAdapter(
            env=env,
            task_id="sort",
            parser=FakeParser(),
            executor=FakeExecutor(env),
        )
        controller = build_controller("open_loop", planner, executor, uncertainty_mode="none")

        row = controller.run_episode(env, "sort block", max_steps=1)

        assert row["success"]
        assert row["completed_subtasks"] == 1
        assert row["failed_subtasks"] == 0
        assert row["planner_calls"] == 1


def test_legacy_rrt_adapter_maps_success_and_fake_uncertainty():
    env = FakeLegacyEnv()
    env.done = True
    adapter = LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id="sort",
        parser=FakeParser(),
        executor=FakeExecutor(env),
    )
    call = SkillCall("ALL", LEGACY_ACTION_PLAN, {"response": make_wait_response(["Alice", "Bob"])})

    adapter.reset(env, ExecutionContext())
    adapter.start_skill(call, env.get_obs())
    feedback = adapter.step(env.get_obs())

    assert feedback.status == BTStatus.SUCCESS
    assert feedback.progress.postcondition_satisfied
    assert feedback.uncertainty.source == "fake_legacy_policy_metadata"
    assert feedback.raw_info["fake_uncertainty"]


def test_legacy_rrt_adapter_maps_parse_failure_to_crie_feedback():
    env = FakeLegacyEnv()
    adapter = LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id="sort",
        parser=FakeParser(success=False, reason="bad response"),
        executor=FakeExecutor(env),
    )
    call = SkillCall("ALL", LEGACY_ACTION_PLAN, {"response": "EXECUTE"})

    adapter.reset(env, ExecutionContext())
    adapter.start_skill(call, env.get_obs())
    feedback = adapter.step(env.get_obs())

    assert feedback.status == BTStatus.FAILURE
    assert feedback.failure.failure_code == FailureCode.POSTCONDITION_FAILED
    assert feedback.raw_info["parse_reason"] == "bad response"
