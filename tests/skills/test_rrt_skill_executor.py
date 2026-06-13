import os

from rocobench.skills.executor import RRTSkillExecutor
from rocobench.skills.models import PreparedSkillExecution, SkillCall, SkillExecutionStatus, SkillPlan


class FakeEnv:
    def __init__(self):
        self.physics = object()
        self.steps = 0

    def get_graspable_objects(self):
        return {"Alice": ["apple"], "Bob": ["apple"]}

    def get_allowed_collision_pairs(self):
        return []

    def step(self, sim_action, verbose=False):
        del sim_action, verbose
        self.steps += 1
        return "obs{}".format(self.steps), 0.5, False, {"is_success": False}


class FakePolicy:
    plan_success = True
    action_count = 2

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.rrt_plan_results = ["rrt"]
        self.action_buffer = ["a{}".format(i) for i in range(self.action_count)]
        self.action_idx = 0

    def plan(self, env):
        del env
        if not self.plan_success:
            return False, "no path"
        return True, "planned"

    @property
    def plan_exhausted(self):
        return self.action_idx >= len(self.action_buffer)

    def act(self, obs, physics):
        del obs, physics
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        return action


class FailingPolicy(FakePolicy):
    plan_success = False


class LongPolicy(FakePolicy):
    action_count = 10


class SpyObserver:
    def __init__(self):
        self.before = []
        self.after = []

    def before_step(self, obs, action, metadata):
        self.before.append((obs, action, dict(metadata)))

    def after_step(self, result):
        self.after.append(dict(result))


def make_plan(prepared=True, backend="rrt"):
    plan = SkillPlan([SkillCall("Alice", "WAIT", {}, "WAIT()")], "EXECUTE")
    if prepared:
        plan.prepared_execution = PreparedSkillExecution(backend, plan.plan_id, ["compiled"])
    return plan


def test_executor_missing_or_wrong_preparation():
    executor = RRTSkillExecutor(FakeEnv(), {}, policy_factory=FakePolicy)

    assert executor.execute(make_plan(prepared=False), "obs").status == SkillExecutionStatus.NOT_PREPARED
    assert executor.execute(make_plan(backend="learned"), "obs").status == SkillExecutionStatus.INVALID_PLAN


def test_executor_motion_planning_failure():
    executor = RRTSkillExecutor(FakeEnv(), {}, policy_factory=FailingPolicy)
    result = executor.execute(make_plan(), "obs")

    assert not result.success
    assert result.status == SkillExecutionStatus.MOTION_PLANNING_FAILED
    assert result.reason == "no path"


def test_executor_success_and_artifacts(tmp_path):
    env = FakeEnv()
    executor = RRTSkillExecutor(env, {}, policy_factory=FakePolicy)
    result = executor.execute(make_plan(), "obs", artifact_dir=str(tmp_path))

    assert result.success
    assert result.status == SkillExecutionStatus.SUCCESS
    assert result.num_sim_steps == 2
    assert os.path.exists(tmp_path / "compiled_path_plans.pkl")
    assert os.path.exists(tmp_path / "rrt_plan_0.pkl")
    assert os.path.exists(tmp_path / "actions_0.pkl")


def test_executor_observer_wraps_exact_step_boundary():
    env = FakeEnv()
    observer = SpyObserver()
    executor = RRTSkillExecutor(env, {}, policy_factory=FakePolicy, transition_observer=observer)

    result = executor.execute(make_plan(), "obs")

    assert result.success
    assert [entry[0] for entry in observer.before] == ["obs", "obs1"]
    assert [entry[1] for entry in observer.before] == ["a0", "a1"]
    assert [entry[2]["skill_step_index"] for entry in observer.before] == [0, 1]
    assert [entry["skill_step_index"] for entry in observer.after] == [1, 2]


def test_executor_timeout():
    executor = RRTSkillExecutor(FakeEnv(), {}, policy_factory=LongPolicy, max_sim_steps=3)
    result = executor.execute(make_plan(), "obs")

    assert not result.success
    assert result.status == SkillExecutionStatus.TIMEOUT
    assert result.num_sim_steps == 3
