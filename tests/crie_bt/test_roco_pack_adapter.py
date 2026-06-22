from rocobench.crie_bt.roco_adapters import (
    FakePackGroceryUncertaintyReporter,
    PackGroceryRRTExecutorAdapter,
    build_pack_grocery_crie_plan,
    crie_call_to_roco_skill_plan,
    pack_grocery_task_spec,
    pack_subtask_success,
)
from rocobench.crie_bt.status import BTStatus, FailureCode
from rocobench.crie_bt.types import ExecutionContext, SkillCall
from rocobench.skills.models import PreparedSkillExecution, SkillExecutionResult, SkillExecutionStatus
from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT, build_pack_grocery_skill_registry
from rocobench.skills.validation import PackGrocerySkillPlanValidator


class FakeObjectState:
    def __init__(self, name, contacts=None, xpos=None):
        self.name = name
        self.contacts = set(contacts or [])
        self.xpos = xpos or (0.0, 0.0, 0.0)


class FakeRobotState:
    def __init__(self, contacts=None, ee_xpos=None):
        self.contacts = set(contacts or [])
        self.ee_xpos = ee_xpos or (0.0, 0.0, 0.25)


class FakeObs:
    def __init__(self):
        self.objects = {
            "apple": FakeObjectState("apple"),
            "banana": FakeObjectState("banana"),
        }
        self.ur5e_robotiq = FakeRobotState()
        self.panda = FakeRobotState()


class FakePackEnv:
    def __init__(self):
        self.item_names = ["apple", "banana"]
        self.bin_slot_xposes = {
            "bin_front_left": (0.0, 0.0, 0.0),
            "bin_front_right": (1.0, 0.0, 0.0),
        }
        self.robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}
        self.robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}
        self.occupancy = {slot_name: None for slot_name in self.bin_slot_xposes}
        self.obs = FakeObs()

    def reset(self):
        return self.obs

    def get_obs(self):
        return self.obs

    def get_agent_held_object(self, obs, agent_name):
        robot_state = getattr(obs, self.robot_name_map_inv[agent_name])
        for item_name in self.item_names:
            if item_name in robot_state.contacts:
                return item_name
        return None

    def get_slot_occupancy(self, obs):
        return dict(self.occupancy)

    def get_packed_slot_for_object(self, obs, object_name):
        del obs
        for slot_name, occupant in self.occupancy.items():
            if occupant == object_name:
                return slot_name
        return None


class FakeCompiler:
    def compile(self, plan, obs):
        del obs
        return PreparedSkillExecution("rrt", plan.plan_id, ["compiled"])


class FakeExecutor:
    def __init__(self, env, success=True, status=SkillExecutionStatus.SUCCESS):
        self.env = env
        self.success = success
        self.status = status

    def execute(self, plan, obs, artifact_dir=None):
        del plan, obs, artifact_dir
        if self.success:
            self.env.occupancy["bin_front_left"] = "apple"
        return SkillExecutionResult(
            success=self.success,
            status=self.status,
            reason="" if self.success else "failed",
            num_sim_steps=3,
            reward=1.0 if self.success else 0.0,
            done=False,
            info={"is_success": self.success},
        )


def _validator(env):
    registry = build_pack_grocery_skill_registry(["Alice", "Bob"])
    return PackGrocerySkillPlanValidator(env, registry, ["Alice", "Bob"])


def test_pack_task_spec_references_existing_pack_grocery_task():
    env = FakePackEnv()
    spec = pack_grocery_task_spec(env)
    assert spec["task"] == "PackGroceryTask"
    assert spec["source"] == "rocobench.envs.task_pack.PackGroceryTask"
    assert PUT_OBJECT_IN_CONTAINER in spec["skills"]
    assert WAIT in spec["skills"]
    assert "RRTSkillExecutor" in spec["skills"][PUT_OBJECT_IN_CONTAINER]["failure_checks"][2]


def test_crie_call_converts_to_existing_roco_skill_plan_with_passive_wait():
    call = SkillCall("Alice", PUT_OBJECT_IN_CONTAINER, {"object": "apple", "container": "bin_front_left"})
    plan = crie_call_to_roco_skill_plan(call, ["Alice", "Bob"], step_id="s1")

    assert [item.agent_name for item in plan.calls] == ["Alice", "Bob"]
    assert plan.calls[0].skill_name == PUT_OBJECT_IN_CONTAINER
    assert plan.calls[1].skill_name == WAIT
    assert "NAME Bob ACTION WAIT()" in plan.parsed_proposal


def test_build_pack_grocery_crie_plan_uses_existing_env_items_and_slots():
    env = FakePackEnv()
    plan = build_pack_grocery_crie_plan(env, env.get_obs(), active_agent="Alice")

    assert [step.skill_call.arguments["object"] for step in plan.steps] == ["apple", "banana"]
    assert plan.steps[0].skill_call.arguments["container"] == "bin_front_left"
    assert plan.metadata["task_spec"]["task"] == "PackGroceryTask"


def test_pack_subtask_success_uses_existing_env_postcondition():
    env = FakePackEnv()
    obs = env.get_obs()
    call = SkillCall("Alice", PUT_OBJECT_IN_CONTAINER, {"object": "apple", "container": "bin_front_left"})

    assert not pack_subtask_success(env, obs, call)
    env.occupancy["bin_front_left"] = "apple"
    assert pack_subtask_success(env, obs, call)


def test_pack_rrt_adapter_maps_validation_failure_to_crie_feedback():
    env = FakePackEnv()
    call = SkillCall("Alice", PUT_OBJECT_IN_CONTAINER, {"object": "apple", "container": "missing_slot"})
    adapter = PackGroceryRRTExecutorAdapter(
        env=env,
        agent_names=["Alice", "Bob"],
        validator=_validator(env),
        compiler=FakeCompiler(),
        executor=FakeExecutor(env),
    )

    adapter.reset(env, ExecutionContext())
    adapter.start_skill(call, env.get_obs())
    feedback = adapter.step(env.get_obs())

    assert feedback.status == BTStatus.FAILURE
    assert feedback.failure.failure_code == FailureCode.WRONG_TARGET
    assert "validation_issues" in feedback.raw_info


def test_pack_rrt_adapter_maps_success_and_fake_uncertainty():
    env = FakePackEnv()
    call = SkillCall("Alice", PUT_OBJECT_IN_CONTAINER, {"object": "apple", "container": "bin_front_left"})
    adapter = PackGroceryRRTExecutorAdapter(
        env=env,
        agent_names=["Alice", "Bob"],
        validator=_validator(env),
        compiler=FakeCompiler(),
        executor=FakeExecutor(env),
        uncertainty_reporter=FakePackGroceryUncertaintyReporter("medium"),
    )

    adapter.reset(env, ExecutionContext())
    adapter.start_skill(call, env.get_obs())
    feedback = adapter.step(env.get_obs())

    assert feedback.status == BTStatus.SUCCESS
    assert feedback.progress.postcondition_satisfied
    assert feedback.uncertainty.source == "fake_pack_grocery_policy_metadata"
    assert feedback.raw_info["fake_uncertainty"]
