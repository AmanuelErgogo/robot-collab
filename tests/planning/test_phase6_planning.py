import json

from prompting.skill_parser import SkillResponseParser
from rocobench.planning import (
    CannedPlanner,
    ExecutorRoute,
    GroundedFeedbackRenderer,
    Phase6PromptRenderer,
    RecoveryAction,
    ReplanningPolicy,
    SequentialPlanningController,
    SkillExecutorRouter,
    build_state_summary,
)
from rocobench.planning.planner_events import PlannerEventType
from rocobench.skills import PackGrocerySkillPlanValidator, build_pack_grocery_skill_registry
from rocobench.skills.models import PreparedSkillExecution, SkillExecutionResult, SkillExecutionStatus


PLAN_APPLE_LEFT = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
"""

PLAN_APPLE_RIGHT = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_right)
NAME Bob ACTION WAIT()
"""

class FakeObjectState(object):
    def __init__(self, name, xpos):
        self.name = name
        self.xpos = xpos
        self.contacts = set()


class FakeRobotState(object):
    def __init__(self, contacts=None):
        self.contacts = set(contacts or [])


class FakeObs(object):
    def __init__(self, positions, held):
        self.objects = {
            name: FakeObjectState(name, xpos)
            for name, xpos in positions.items()
        }
        self.ur5e_robotiq = FakeRobotState([held["Alice"]] if held.get("Alice") else [])
        self.panda = FakeRobotState([held["Bob"]] if held.get("Bob") else [])


class FakePackEnv(object):
    def __init__(self):
        self.item_names = ["apple", "banana", "milk"]
        self.bin_slot_xposes = {
            "bin_front_left": (0.0, 0.0, 0.0),
            "bin_front_right": (1.0, 0.0, 0.0),
            "bin_back_left": (0.0, 1.0, 0.0),
        }
        self.robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}
        self.robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}
        self.positions = {
            "apple": (0.0, 0.0, 0.1),
            "banana": (0.2, 0.0, 0.1),
            "milk": (0.4, 0.0, 0.1),
        }
        self.held = {"Alice": None, "Bob": None}
        self.occupancy = {slot_name: None for slot_name in self.bin_slot_xposes}
        self.restore_count = 0

    def get_obs(self):
        return FakeObs(dict(self.positions), dict(self.held))

    def describe_obs(self, obs):
        del obs
        return "fake pack scene"

    def get_agent_held_object(self, obs, agent_name):
        del obs
        return self.held.get(agent_name)

    def get_slot_occupancy(self, obs):
        del obs
        return dict(self.occupancy)

    def get_packed_slot_for_object(self, obs, object_name):
        del obs
        for slot_name, occupant in self.occupancy.items():
            if occupant == object_name:
                return slot_name
        return None

    def snapshot(self):
        return {
            "positions": dict(self.positions),
            "held": dict(self.held),
            "occupancy": dict(self.occupancy),
        }

    def restore(self, snapshot):
        self.positions = dict(snapshot["positions"])
        self.held = dict(snapshot["held"])
        self.occupancy = dict(snapshot["occupancy"])
        self.restore_count += 1

    def get_state_digest(self):
        payload = {
            "positions": self.positions,
            "held": self.held,
            "occupancy": self.occupancy,
        }
        return json.dumps(payload, sort_keys=True)

    def place_from_plan(self, plan):
        for call in plan.calls:
            if call.skill_name != "WAIT":
                obj = call.arguments["object"]
                target = call.arguments["container"]
                self.occupancy[target] = obj
                self.positions[obj] = self.bin_slot_xposes[target]


class FakeCompiler(object):
    def __init__(self):
        self.compile_count = 0

    def compile(self, plan, obs):
        del obs
        self.compile_count += 1
        return PreparedSkillExecution("rrt", plan.plan_id, ["compiled"])


class FakeExecutor(object):
    def __init__(self, env, outcomes):
        self.env = env
        self.outcomes = list(outcomes)
        self.calls = []

    def execute(self, plan, obs, artifact_dir=None):
        self.calls.append({"plan": plan, "obs": obs, "artifact_dir": artifact_dir})
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = {"success": True, "is_success": True}
        if outcome.get("mutate"):
            outcome["mutate"](self.env, plan)
        if outcome.get("success"):
            self.env.place_from_plan(plan)
            return SkillExecutionResult(
                success=True,
                status=SkillExecutionStatus.SUCCESS,
                reason="",
                num_sim_steps=int(outcome.get("steps", 1)),
                reward=1.0 if outcome.get("is_success", False) else 0.0,
                done=bool(outcome.get("is_success", False)),
                info={"is_success": bool(outcome.get("is_success", False))},
                metadata={"learned_success": outcome.get("learned_success", False)},
            )
        return SkillExecutionResult(
            success=False,
            status=outcome.get("status", SkillExecutionStatus.EXECUTION_FAILED),
            reason=outcome.get("reason", outcome.get("failure_code", "failed")),
            num_sim_steps=int(outcome.get("steps", 1)),
            reward=0.0,
            done=False,
            info={"is_success": False},
            metadata={
                "failure_code": outcome.get("failure_code"),
                "progress_stage": outcome.get("progress_stage", "not_started"),
            },
        )


def make_controller(env, learned_outcomes=None, rrt_outcomes=None, policy=None, routes=None):
    registry = build_pack_grocery_skill_registry(["Alice", "Bob"])
    parser = SkillResponseParser(registry, ["Alice", "Bob"])
    validator = PackGrocerySkillPlanValidator(env, registry, ["Alice", "Bob"])
    router = SkillExecutorRouter(
        routes=routes if routes is not None else [
            ExecutorRoute(
                agent="Alice",
                skill="PUT_OBJECT_IN_CONTAINER",
                primary="learned",
                fallback="rrt",
                policy_id="internal_policy_id",
            )
        ],
        default_backend="rrt",
    )
    compiler = FakeCompiler()
    learned = FakeExecutor(env, learned_outcomes or [{"success": True, "is_success": True, "learned_success": True}])
    rrt = FakeExecutor(env, rrt_outcomes or [{"success": True, "is_success": True}])
    controller = SequentialPlanningController(
        env=env,
        agent_names=["Alice", "Bob"],
        parser=parser,
        validator=validator,
        router=router,
        policy=policy or ReplanningPolicy(max_plan_rounds=4),
        executors={"learned": learned, "rrt": rrt},
        rrt_compiler=compiler,
        registry=registry,
    )
    return controller, learned, rrt, compiler, registry, router


def test_router_deterministic_and_prompt_hides_policy_id():
    env = FakePackEnv()
    _, _, _, _, registry, router = make_controller(env)
    parser = SkillResponseParser(registry, ["Alice", "Bob"])
    ok, _, plans = parser.parse(env.get_obs(), PLAN_APPLE_LEFT)
    assert ok

    decision = router.resolve(plans[0].calls[0])

    assert decision.backend == "learned"
    assert decision.policy_id == "internal_policy_id"
    prompt = router.prompt_capabilities(["Alice", "Bob"], registry)
    assert "learned capability available" in prompt
    assert "internal_policy_id" not in prompt
    assert "checkpoint" not in prompt.lower()


def test_prompt_and_feedback_render_current_state_with_separate_inference():
    env = FakePackEnv()
    controller, _, _, _, registry, router = make_controller(env)
    obs = env.get_obs()
    parser = SkillResponseParser(registry, ["Alice", "Bob"])
    _, _, plans = parser.parse(obs, PLAN_APPLE_LEFT)
    summary = build_state_summary(env, obs, ["Alice", "Bob"], plans[0].calls[0], inferred_facts=["Likely slipped during transport."])

    feedback = GroundedFeedbackRenderer().render_failure(
        plans[0].calls[0],
        "execution_failed",
        "SLIPPAGE",
        "transporting",
        summary,
        controller.budget,
        RecoveryAction.REPLAN_FROM_CURRENT_STATE,
    )
    prompt = Phase6PromptRenderer(router, registry, ["Alice", "Bob"]).render(summary, feedback, controller.budget)

    assert "Measured current state:" in feedback
    assert "Inferred explanation:" in feedback
    assert "Likely slipped" in feedback
    assert "Do not output low-level paths" in prompt
    assert "EXECUTE" in prompt


def test_controller_learned_success_records_separate_metrics_and_events():
    env = FakePackEnv()
    controller, _, _, _, _, _ = make_controller(env)
    result = controller.run(CannedPlanner([PLAN_APPLE_LEFT]), initial_obs=env.get_obs())

    assert result.success is True
    assert result.metrics.learned_attempts == 1
    assert result.metrics.learned_successes == 1
    assert result.metrics.fallback_attempts == 0
    event_types = result.events.event_types()
    assert event_types[:4] == [
        PlannerEventType.OBSERVED.value,
        PlannerEventType.PROMPTED.value,
        PlannerEventType.PLAN_PARSED.value,
        PlannerEventType.EXECUTOR_SELECTED.value,
    ]
    assert PlannerEventType.SKILL_SUCCEEDED.value in event_types
    assert event_types[-1] == PlannerEventType.TASK_SUCCEEDED.value


def test_invalid_plan_replans_with_grounded_feedback():
    env = FakePackEnv()
    env.occupancy["bin_front_left"] = "banana"
    policy = ReplanningPolicy(max_plan_rounds=2)
    controller, _, _, _, _, _ = make_controller(env, policy=policy)
    planner = CannedPlanner([PLAN_APPLE_LEFT, PLAN_APPLE_RIGHT])
    result = controller.run(planner, initial_obs=env.get_obs())

    assert result.success is True
    assert result.metrics.invalid_plans == 1
    assert "TARGET_OCCUPIED" in result.feedback_history[0]
    assert "bin_front_left is occupied by banana" in result.feedback_history[0]
    assert len(planner.prompts) == 2
    assert "TARGET_OCCUPIED" in planner.prompts[1]


def test_missed_grasp_retries_same_learned_plan_without_new_prompt():
    env = FakePackEnv()
    controller, learned, _, _, _, _ = make_controller(
        env,
        learned_outcomes=[
            {"success": False, "failure_code": "MISSED_GRASP", "reason": "missed"},
            {"success": True, "is_success": True, "learned_success": True},
        ],
        policy=ReplanningPolicy(max_plan_rounds=3, max_retries_per_skill=1, max_learned_failures=3),
    )
    planner = CannedPlanner([PLAN_APPLE_LEFT])
    result = controller.run(planner, initial_obs=env.get_obs())

    assert result.success is True
    assert result.metrics.learned_attempts == 2
    assert len(learned.calls) == 2
    assert len(planner.prompts) == 1
    assert "MISSED_GRASP" in result.feedback_history[0]
    assert result.budget.retries_by_key


def test_slippage_replans_from_failed_current_state_without_rollback():
    env = FakePackEnv()

    def slip(env, plan):
        del plan
        env.positions["apple"] = (9.0, 9.0, 0.1)

    controller, _, _, _, _, _ = make_controller(
        env,
        learned_outcomes=[
            {"success": False, "failure_code": "SLIPPAGE", "reason": "slip", "progress_stage": "transporting", "mutate": slip},
            {"success": True, "is_success": True, "learned_success": True},
        ],
        policy=ReplanningPolicy(max_plan_rounds=3),
    )
    result = controller.run(CannedPlanner([PLAN_APPLE_LEFT, PLAN_APPLE_RIGHT]), initial_obs=env.get_obs())

    assert result.success is True
    assert "apple position is (9.000, 9.000, 0.100)." in result.feedback_history[0]
    assert PlannerEventType.STATE_RESTORED.value not in result.events.event_types()


def test_inference_failure_rolls_back_then_uses_explicit_rrt_fallback():
    env = FakePackEnv()

    def corrupt_state(env, plan):
        del plan
        env.positions["apple"] = (5.0, 5.0, 0.1)

    controller, _, rrt, _, _, _ = make_controller(
        env,
        learned_outcomes=[
            {"success": False, "failure_code": "POLICY_INFERENCE_FAILURE", "reason": "forced inference", "mutate": corrupt_state}
        ],
        rrt_outcomes=[{"success": True, "is_success": True}],
        policy=ReplanningPolicy(max_plan_rounds=2, max_fallbacks=1),
    )
    result = controller.run(CannedPlanner([PLAN_APPLE_LEFT]), initial_obs=env.get_obs())

    assert result.success is True
    assert env.restore_count == 1
    assert len(rrt.calls) == 1
    assert result.metrics.fallback_attempts == 1
    assert result.metrics.fallback_successes == 1
    assert env.positions["apple"] == env.bin_slot_xposes["bin_front_left"]
    restore_events = [event for event in result.events.to_list() if event["event_type"] == PlannerEventType.STATE_RESTORED.value]
    assert restore_events[0]["metadata"]["equal"] is True
    assert PlannerEventType.FALLBACK_STARTED.value in result.events.event_types()


def test_repeated_failure_fingerprint_forces_fallback():
    env = FakePackEnv()
    controller, _, rrt, _, _, _ = make_controller(
        env,
        learned_outcomes=[
            {"success": False, "failure_code": "MISSED_GRASP", "reason": "missed"},
            {"success": False, "failure_code": "MISSED_GRASP", "reason": "missed again"},
        ],
        rrt_outcomes=[{"success": True, "is_success": True}],
        policy=ReplanningPolicy(max_plan_rounds=3, max_retries_per_skill=3, max_repeated_failures=1, max_fallbacks=1),
    )
    result = controller.run(CannedPlanner([PLAN_APPLE_LEFT]), initial_obs=env.get_obs())

    assert result.success is True
    assert len(rrt.calls) == 1
    failed_events = [event for event in result.events.to_list() if event["event_type"] == PlannerEventType.EXECUTION_FAILED.value]
    assert failed_events[-1]["metadata"]["recovery"]["fingerprint_count"] == 2
    assert "Repeated failure fingerprint" in failed_events[-1]["metadata"]["recovery"]["reason"]


def test_budget_exhaustion_terminates_without_live_llm():
    env = FakePackEnv()
    controller, _, _, _, _, _ = make_controller(env, policy=ReplanningPolicy(max_plan_rounds=1))
    result = controller.run(CannedPlanner(["EXECUTE\nNAME Alice ACTION WAIT()\n"]), initial_obs=env.get_obs())

    assert result.success is False
    assert result.reason == "Plan round budget exhausted."
    assert result.events.event_types()[-1] == PlannerEventType.BUDGET_EXHAUSTED.value


def test_rrt_only_mode_remains_available():
    env = FakePackEnv()
    route = ExecutorRoute(agent="Alice", skill="PUT_OBJECT_IN_CONTAINER", primary="rrt")
    controller, learned, rrt, compiler, _, _ = make_controller(env, routes=[route])
    result = controller.run(CannedPlanner([PLAN_APPLE_LEFT]), initial_obs=env.get_obs())

    assert result.success is True
    assert len(learned.calls) == 0
    assert len(rrt.calls) == 1
    assert compiler.compile_count == 1
    assert result.metrics.rrt_attempts == 1
    assert result.metrics.learned_attempts == 0
