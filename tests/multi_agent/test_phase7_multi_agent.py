import time

import numpy as np

from rocobench.multi_agent import (
    AgentActionFragment,
    CentralJointStepper,
    CentralSafetyMonitor,
    JointActionMergeError,
    MultiAgentScheduler,
    ScheduleDecision,
    ScheduleGroup,
    ScheduleMode,
    SchedulerConfig,
    SequentialMultiAgentExecutor,
    SynchronizedExecutor,
)
from rocobench.skills.models import SkillCall, SkillExecutionResult, SkillExecutionStatus, SkillPlan
from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT


def put(agent, obj, target):
    raw = "PUT_OBJECT_IN_CONTAINER(object={}, container={})".format(obj, target)
    return SkillCall(
        agent_name=agent,
        skill_name=PUT_OBJECT_IN_CONTAINER,
        arguments={"object": obj, "container": target},
        raw_action=raw,
    )


def wait(agent):
    return SkillCall(agent_name=agent, skill_name=WAIT, arguments={}, raw_action="WAIT()")


def skill_plan(*calls):
    return SkillPlan(list(calls), "EXECUTE")


class FakeSimAction(object):
    def __init__(
        self,
        ctrl_idxs,
        ctrl_vals,
        qpos_idxs,
        qpos_target,
        eq_active_idxs=None,
        eq_active_vals=None,
    ):
        self.ctrl_idxs = ctrl_idxs
        self.ctrl_vals = ctrl_vals
        self.qpos_idxs = qpos_idxs
        self.qpos_target = qpos_target
        self.eq_active_idxs = eq_active_idxs
        self.eq_active_vals = eq_active_vals


class FakeJointEnv(object):
    def __init__(self, info=None, done=False):
        self.info = dict(info or {})
        self.done = bool(done)
        self.step_count = 0
        self.actions = []

    def step(self, sim_action, verbose=False):
        del verbose
        self.step_count += 1
        self.actions.append(sim_action)
        return {"step": self.step_count}, 1.0, self.done, dict(self.info)


class StaticProvider(object):
    def __init__(self, fragment, delay_s=0.0):
        self.fragment = fragment
        self.delay_s = float(delay_s)
        self.reset_count = 0
        self.calls = 0

    def reset(self):
        self.reset_count += 1

    def next_action(self, obs, agent):
        del obs, agent
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.fragment


class FakeSkillExecutor(object):
    backend_name = "fake"

    def __init__(self, success=True, steps=1):
        self.success = bool(success)
        self.steps = int(steps)
        self.reset_agents = []
        self.executed_plans = []

    def reset_agent(self, agent_name):
        self.reset_agents.append(agent_name)

    def execute(self, plan, obs, artifact_dir=None):
        del obs, artifact_dir
        self.executed_plans.append(plan)
        status = SkillExecutionStatus.SUCCESS if self.success else SkillExecutionStatus.EXECUTION_FAILED
        return SkillExecutionResult(
            success=self.success,
            status=status,
            reason="" if self.success else "forced failure",
            num_sim_steps=self.steps,
            reward=1.0 if self.success else 0.0,
            done=self.success,
            info={"is_success": self.success},
            metadata={"executed": True},
        )


def fragment(agent, ctrl_idx, ctrl_val, qpos_idx, qpos_val):
    return AgentActionFragment(
        agent_name=agent,
        ctrl_idxs=np.asarray([ctrl_idx], dtype=np.int32),
        ctrl_vals=np.asarray([ctrl_val], dtype=np.float32),
        qpos_idxs=np.asarray([qpos_idx], dtype=np.int32),
        qpos_target=np.asarray([qpos_val], dtype=np.float32),
    )


def concurrent_schedule(calls):
    group = ScheduleGroup(0, tuple(calls), ScheduleMode.CONCURRENT.value)
    return ScheduleDecision(
        mode=ScheduleMode.CONCURRENT.value,
        ordered_groups=(group,),
        resource_claims={},
        rule_id="test.concurrent",
        reason="test schedule",
    )


def test_sequential_scheduler_is_deterministic_and_routes_per_agent():
    plan = skill_plan(
        put("Bob", "banana", "bin_front_right"),
        put("Alice", "apple", "bin_front_left"),
    )

    decision = MultiAgentScheduler().schedule(plan)

    assert decision.mode == ScheduleMode.SEQUENTIAL.value
    assert [group.calls[0].agent_name for group in decision.ordered_groups] == ["Alice", "Bob"]
    assert "object:apple" in decision.resource_claims["Alice"].exclusive
    assert "object:banana" in decision.resource_claims["Bob"].exclusive


def test_scheduler_rejects_same_object_and_same_target_conflicts():
    scheduler = MultiAgentScheduler(SchedulerConfig(enable_concurrency=True))

    same_object = scheduler.schedule(
        skill_plan(put("Alice", "apple", "bin_front_left"), put("Bob", "apple", "bin_front_right"))
    )
    same_target = scheduler.schedule(
        skill_plan(put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_left"))
    )

    assert same_object.mode == ScheduleMode.REJECT.value
    assert any("object:apple" in conflict for conflict in same_object.conflicts)
    assert same_target.mode == ScheduleMode.REJECT.value
    assert any("target:bin_front_left" in conflict for conflict in same_target.conflicts)


def test_scheduler_rejects_overlapping_workspace_for_concurrency():
    scheduler = MultiAgentScheduler(SchedulerConfig(enable_concurrency=True))

    decision = scheduler.schedule(
        skill_plan(put("Alice", "apple", "bin_front_right"), put("Bob", "banana", "bin_front_left"))
    )

    assert decision.mode == ScheduleMode.REJECT.value
    assert any("overlap workspace" in conflict for conflict in decision.conflicts)


def test_scheduler_admits_disjoint_concurrency_when_feature_enabled():
    scheduler = MultiAgentScheduler(SchedulerConfig(enable_concurrency=True))

    decision = scheduler.schedule(
        skill_plan(put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_right"))
    )

    assert decision.mode == ScheduleMode.CONCURRENT.value
    assert len(decision.ordered_groups) == 1
    assert [call.agent_name for call in decision.ordered_groups[0].calls] == ["Alice", "Bob"]


def test_all_wait_plan_is_rejected():
    decision = MultiAgentScheduler().schedule(skill_plan(wait("Alice"), wait("Bob")))

    assert decision.mode == ScheduleMode.REJECT.value
    assert decision.rule_id == "phase7.no_active_skill"


def test_sequential_executor_resets_agents_and_combines_failure():
    plan = skill_plan(put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_right"))
    schedule = MultiAgentScheduler().schedule(plan)
    alice = FakeSkillExecutor(success=True, steps=2)
    bob = FakeSkillExecutor(success=False, steps=3)

    result = SequentialMultiAgentExecutor(
        ["Alice", "Bob"],
        {"Alice": alice, "Bob": bob},
    ).execute(schedule, plan, obs={"fake": True})

    assert result.success is False
    assert result.central_status == "FAILED"
    assert alice.reset_agents == ["Alice"]
    assert bob.reset_agents == ["Bob"]
    assert alice.executed_plans[0].calls[1].skill_name == WAIT
    assert bob.executed_plans[0].calls[0].skill_name == WAIT
    assert result.per_agent_outcomes["Alice"].success is True
    assert result.per_agent_outcomes["Bob"].success is False
    assert result.metrics.makespan_steps == 5


def test_joint_stepper_merges_fragments_and_steps_once():
    env = FakeJointEnv()
    stepper = CentralJointStepper(env, ["Alice", "Bob"], sim_action_factory=FakeSimAction)

    result = stepper.step(
        {
            "Bob": fragment("Bob", 3, 0.3, 13, 1.3),
            "Alice": fragment("Alice", 1, 0.1, 11, 1.1),
        }
    )

    assert env.step_count == 1
    assert result.step_index == 1
    action = env.actions[0]
    assert action.ctrl_idxs.tolist() == [1, 3]
    assert np.allclose(action.ctrl_vals, [0.1, 0.3])
    assert action.qpos_idxs.tolist() == [11, 13]
    assert np.allclose(action.qpos_target, [1.1, 1.3])


def test_joint_stepper_rejects_conflicting_control_index():
    stepper = CentralJointStepper(FakeJointEnv(), ["Alice", "Bob"], sim_action_factory=FakeSimAction)

    try:
        stepper.merge(
            {
                "Alice": fragment("Alice", 1, 0.1, 11, 1.1),
                "Bob": fragment("Bob", 1, 0.2, 13, 1.3),
            }
        )
    except JointActionMergeError as exc:
        assert "ctrl index 1" in str(exc)
    else:
        raise AssertionError("Expected control-index conflict")


def test_synchronized_executor_stop_all_on_nonfinite_action():
    env = FakeJointEnv()
    alice = fragment("Alice", 1, float("nan"), 11, 1.1)
    bob = fragment("Bob", 3, 0.3, 13, 1.3)
    schedule = concurrent_schedule([put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_right")])

    result = SynchronizedExecutor(
        env,
        ["Alice", "Bob"],
        {"Alice": StaticProvider(alice), "Bob": StaticProvider(bob)},
        sim_action_factory=FakeSimAction,
    ).run(schedule, initial_obs={}, max_joint_steps=2)

    assert result.success is False
    assert result.central_status == "STOP_ALL"
    assert env.step_count == 0
    assert [event.code for event in result.safety_events] == ["NONFINITE_ACTION"]


def test_synchronized_executor_stop_all_on_policy_latency():
    env = FakeJointEnv()
    schedule = concurrent_schedule([put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_right")])

    result = SynchronizedExecutor(
        env,
        ["Alice", "Bob"],
        {
            "Alice": StaticProvider(fragment("Alice", 1, 0.1, 11, 1.1), delay_s=0.002),
            "Bob": StaticProvider(fragment("Bob", 3, 0.3, 13, 1.3)),
        },
        safety_monitor=CentralSafetyMonitor(max_policy_latency_ms=0.0),
        sim_action_factory=FakeSimAction,
    ).run(schedule, initial_obs={}, max_joint_steps=1)

    assert result.success is False
    assert result.central_status == "STOP_ALL"
    assert env.step_count == 0
    assert any(event.code == "POLICY_LATENCY_EXCEEDED" for event in result.safety_events)


def test_synchronized_executor_stop_all_on_forbidden_contact_info():
    env = FakeJointEnv(info={"collision": True})
    schedule = concurrent_schedule([put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_right")])

    result = SynchronizedExecutor(
        env,
        ["Alice", "Bob"],
        {
            "Alice": StaticProvider(fragment("Alice", 1, 0.1, 11, 1.1)),
            "Bob": StaticProvider(fragment("Bob", 3, 0.3, 13, 1.3)),
        },
        sim_action_factory=FakeSimAction,
    ).run(schedule, initial_obs={}, max_joint_steps=2)

    assert result.success is False
    assert result.central_status == "STOP_ALL"
    assert env.step_count == 1
    assert any(event.code == "FORBIDDEN_CONTACT" for event in result.safety_events)
    assert result.fallback_to_sequential_recommended is True
