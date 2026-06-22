"""Adapters from CRIE-BT contracts to existing RoCo PackGrocery skills."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rocobench.skills.models import (
    SkillCall as RoCoSkillCall,
    SkillExecutionResult,
    SkillExecutionStatus,
    SkillPlan as RoCoSkillPlan,
)
from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT
from rocobench.skills.registry import SkillRegistry

from .executor import BaseSkillExecutor
from .planner import BasePlanner
from .status import BTStatus, FailureCode, ProgressStage
from .types import (
    CollaborativePlan,
    ExecutionContext,
    ExecutionFeedback,
    FailureState,
    PlanStep,
    ProgressState,
    SkillCall,
    SubtaskCommand,
    UncertaintyState,
)


PACK_GROCERY_LOW_LEVEL_ACTIONS = (
    "PICK_OR_PLACE",
    "RRT_PLAN",
    "SIM_ACTION_STEPS",
    "POSTCONDITION_CHECK",
)


def pack_grocery_task_spec(env: Any = None, agent_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return a JSON-friendly CRIE-BT contract for existing PackGroceryTask."""

    if agent_names is None:
        if env is not None:
            agent_names = list(getattr(env, "robot_name_map", {}).values())
        else:
            agent_names = ["Alice", "Bob"]
    item_names = list(getattr(env, "item_names", [])) if env is not None else []
    slot_names = list(getattr(env, "bin_slot_xposes", {}).keys()) if env is not None else []
    return {
        "task": "PackGroceryTask",
        "source": "rocobench.envs.task_pack.PackGroceryTask",
        "skills": {
            PUT_OBJECT_IN_CONTAINER: {
                "arguments": ["object", "container"],
                "description": (
                    "Existing PackGrocery skill: pick the grocery object if needed, "
                    "transport it, place it into a named bin slot, and release it."
                ),
                "supported_agents": list(agent_names),
                "resources": ["agent", "object", "container"],
                "subtask_success": "env.get_packed_slot_for_object(obs, object) == container",
                "failure_checks": [
                    "PackGrocerySkillPlanValidator rejects object/target/agent/resource claims",
                    "RRTSkillCompiler cannot compile legacy action plan",
                    "RRTSkillExecutor reports motion planning failure",
                    "RRTSkillExecutor reports timeout or execution failure",
                    "postcondition check fails after executor success",
                ],
            },
            WAIT: {
                "arguments": [],
                "description": "Existing PackGrocery passive-agent hold skill.",
                "supported_agents": list(agent_names),
                "resources": ["agent"],
                "subtask_success": "agent remains passive for the synchronized skill step",
                "failure_checks": [],
            },
        },
        "subtasks": [
            {
                "name": "pack_grocery_object",
                "skill": PUT_OBJECT_IN_CONTAINER,
                "low_level_actions": list(PACK_GROCERY_LOW_LEVEL_ACTIONS),
                "progress_source": "PackGroceryTask EnvState plus executor feedback",
            }
        ],
        "actions": {
            "PICK_OR_PLACE": "RRTSkillCompiler emits PICK ... PLACE ... or PLACE ... from current held-object state.",
            "RRT_PLAN": "Existing PlannedPathPolicy plans path actions for the compiled skill.",
            "SIM_ACTION_STEPS": "Existing RRTSkillExecutor steps PackGroceryTask with SimAction objects.",
            "POSTCONDITION_CHECK": "CRIE-BT verifies the object is packed in the requested bin slot.",
            WAIT: "Passive agents receive WAIT through the existing skill plan.",
        },
        "observation": {
            "type": "rocobench.envs.base_env.EnvState",
            "objects": "obs.objects[item].xpos and contacts",
            "robots": "obs.<robot_name>.ee_xpos and contacts",
            "packed_slot": "env.get_packed_slot_for_object(obs, object)",
            "slot_occupancy": "env.get_slot_occupancy(obs)",
            "held_object": "env.get_agent_held_object(obs, agent)",
        },
        "objects": item_names,
        "containers": slot_names,
        "uncertainty": {
            "source": "fake_pack_grocery_policy_metadata",
            "fields": ["confidence", "entropy", "action_norm", "chunk_disagreement"],
            "note": "Deterministic placeholder metadata; learned executors can replace it with entropy or ensemble estimators.",
        },
    }


def crie_skill_to_roco_call(skill_call: SkillCall) -> RoCoSkillCall:
    """Convert a CRIE-BT skill call to the existing RoCo skill call type."""

    return RoCoSkillCall(
        agent_name=skill_call.agent,
        skill_name=skill_call.skill_name,
        arguments={str(key): str(value) for key, value in dict(skill_call.arguments).items()},
        raw_action=skill_call.instruction or _render_roco_action(skill_call),
    )


def _render_roco_action(skill_call: SkillCall) -> str:
    args = ", ".join("{}={}".format(key, value) for key, value in sorted(dict(skill_call.arguments).items()))
    return "{}({})".format(skill_call.skill_name, args)


def plan_step_to_roco_skill_plan(
    step: PlanStep,
    agent_names: Sequence[str],
    plan_id: Optional[str] = None,
) -> RoCoSkillPlan:
    """Build a one-active-agent RoCo SkillPlan for a CRIE-BT plan step."""

    calls = []  # type: List[RoCoSkillCall]
    active_agent = step.skill_call.agent
    for agent_name in agent_names:
        if agent_name == active_agent:
            calls.append(crie_skill_to_roco_call(step.skill_call))
        else:
            calls.append(RoCoSkillCall(agent_name, WAIT, {}, "WAIT()"))
    parsed_proposal = "EXECUTE\n" + "\n".join(
        "NAME {} ACTION {}".format(call.agent_name, str(call)) for call in calls
    )
    return RoCoSkillPlan(calls, parsed_proposal, plan_id=plan_id or step.step_id)


def crie_call_to_roco_skill_plan(
    skill_call: SkillCall,
    agent_names: Sequence[str],
    step_id: str = "crie_step",
) -> RoCoSkillPlan:
    step = PlanStep(step_id=step_id, skill_call=skill_call)
    return plan_step_to_roco_skill_plan(step, agent_names, plan_id=step_id)


def pack_subtask_success(env: Any, obs: Any, skill_call: SkillCall) -> bool:
    """Check the existing PackGrocery postcondition for a CRIE-BT subtask."""

    if skill_call.skill_name == WAIT:
        return True
    object_name = skill_call.arguments.get("object")
    container = skill_call.arguments.get("container")
    if not object_name or not container or env is None:
        return False
    if hasattr(env, "get_packed_slot_for_object"):
        return env.get_packed_slot_for_object(obs, object_name) == container
    return False


def first_unpacked_pack_object(env: Any, obs: Any) -> Optional[str]:
    for object_name in list(getattr(env, "item_names", [])):
        if hasattr(env, "get_packed_slot_for_object") and env.get_packed_slot_for_object(obs, object_name) is None:
            return object_name
    return None


def first_free_pack_slot(env: Any, obs: Any) -> Optional[str]:
    if not hasattr(env, "get_slot_occupancy"):
        slots = list(getattr(env, "bin_slot_xposes", {}).keys())
        return slots[0] if slots else None
    occupancy = env.get_slot_occupancy(obs)
    for slot_name in list(getattr(env, "bin_slot_xposes", {}).keys()):
        if occupancy.get(slot_name) is None:
            return slot_name
    return None


def build_pack_grocery_crie_plan(
    env: Any,
    obs: Any,
    object_targets: Optional[Mapping[str, str]] = None,
    active_agent: str = "Alice",
    plan_id: str = "pack_grocery_crie_plan",
    task_goal: str = "Pack groceries into the bin.",
) -> CollaborativePlan:
    """Build a CRIE-BT plan over the existing PackGrocery objects and slots."""

    targets = dict(object_targets or {})
    if not targets:
        free_slots = list(getattr(env, "bin_slot_xposes", {}).keys())
        for index, object_name in enumerate(list(getattr(env, "item_names", []))):
            if index < len(free_slots):
                targets[object_name] = free_slots[index]

    steps = []
    for object_name in list(getattr(env, "item_names", [])):
        container = targets.get(object_name)
        if not container:
            continue
        if hasattr(env, "get_packed_slot_for_object") and env.get_packed_slot_for_object(obs, object_name) == container:
            continue
        skill_call = SkillCall(
            agent=active_agent,
            skill_name=PUT_OBJECT_IN_CONTAINER,
            arguments={"object": object_name, "container": container},
            instruction="Put {} into {}.".format(object_name, container),
        )
        subtask = SubtaskCommand(
            subtask_id="pack_{}".format(object_name),
            skill_call=skill_call,
            goal="Pack {} into {}.".format(object_name, container),
            metadata={
                "task": "PackGroceryTask",
                "success_condition": "env.get_packed_slot_for_object(obs, '{}') == '{}'".format(object_name, container),
                "low_level_actions": list(PACK_GROCERY_LOW_LEVEL_ACTIONS),
            },
        )
        steps.append(
            PlanStep(
                step_id=subtask.subtask_id,
                skill_call=subtask.skill_call,
                role_assignment={active_agent: "active_robot", "Human": "observer"},
                explanation=subtask.goal,
                metadata=subtask.metadata,
            )
        )
    return CollaborativePlan(
        steps=steps,
        plan_id=plan_id,
        task_goal=task_goal,
        metadata={"task_spec": pack_grocery_task_spec(env), "object_targets": targets},
    )


class PackGroceryCRIEPlanner(BasePlanner):
    """Planner over the existing PackGroceryTask item and bin-slot state."""

    def __init__(
        self,
        env: Any,
        object_targets: Optional[Mapping[str, str]] = None,
        active_agent: str = "Alice",
    ) -> None:
        self.env = env
        self.object_targets = dict(object_targets or {})
        self.active_agent = active_agent
        self.calls = 0

    def generate_plan(
        self,
        task_goal: str,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        context: Optional[ExecutionContext] = None,
    ) -> CollaborativePlan:
        del context
        self.calls += 1
        targets = dict(self.object_targets)
        if feedback is not None and feedback.failure.failure_code == FailureCode.TARGET_OCCUPIED:
            object_name = feedback.skill_call.arguments.get("object")
            replacement = first_free_pack_slot(self.env, observation)
            if object_name and replacement:
                targets[object_name] = replacement
                self.object_targets[object_name] = replacement
        return build_pack_grocery_crie_plan(
            env=self.env,
            obs=observation,
            object_targets=targets,
            active_agent=self.active_agent,
            plan_id="pack_grocery_crie_plan_{:03d}".format(self.calls),
            task_goal=task_goal,
        )


class FakePackGroceryUncertaintyReporter(object):
    """Deterministic placeholder for later learned-model uncertainty metadata."""

    def __init__(self, profile: str = "nominal") -> None:
        self.profile = str(profile or "nominal")

    def report(self, result: Optional[SkillExecutionResult], success: bool, failure: FailureState) -> Dict[str, Any]:
        if self.profile == "low_confidence":
            confidence = 0.42
        elif self.profile == "medium":
            confidence = 0.58
        elif failure.is_failure or not success:
            confidence = 0.22
        else:
            confidence = 0.82
        num_steps = int(getattr(result, "num_sim_steps", 0) if result is not None else 0)
        entropy = max(0.0, min(1.0, 1.0 - confidence))
        return {
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "action_norm": round(float(num_steps) / 100.0, 4),
            "chunk_disagreement": round(entropy * 0.5, 4),
            "uncertainty_source": "fake_pack_grocery_policy_metadata",
            "fake_uncertainty": True,
        }


class PackGroceryRRTExecutorAdapter(BaseSkillExecutor):
    """CRIE-BT executor adapter for the existing PackGrocery RRT skill stack."""

    def __init__(
        self,
        env: Any,
        agent_names: Sequence[str],
        validator: Any,
        compiler: Any,
        executor: Any,
        artifact_dir: Optional[str] = None,
        uncertainty_reporter: Optional[FakePackGroceryUncertaintyReporter] = None,
    ) -> None:
        self.env = env
        self.agent_names = list(agent_names)
        self.validator = validator
        self.compiler = compiler
        self.executor = executor
        self.artifact_dir = artifact_dir
        self.uncertainty_reporter = uncertainty_reporter or FakePackGroceryUncertaintyReporter()
        self.context = ExecutionContext()
        self.active_skill = None  # type: Optional[SkillCall]
        self.pending_feedback = None  # type: Optional[ExecutionFeedback]

    def reset(self, env, context: ExecutionContext) -> None:
        if env is not None:
            self.env = env
        self.context = context
        self.active_skill = None
        self.pending_feedback = None

    def start_skill(self, skill_call: SkillCall, observation: Any) -> None:
        self.active_skill = skill_call
        self.pending_feedback = self._run_skill(skill_call, observation)

    def step(self, observation: Any) -> ExecutionFeedback:
        del observation
        if self.pending_feedback is None:
            if self.active_skill is None:
                raise RuntimeError("PackGroceryRRTExecutorAdapter.step called before start_skill.")
            return self._failure_feedback(
                self.active_skill,
                FailureCode.UNKNOWN,
                "No PackGrocery execution feedback was prepared.",
                {},
            )
        feedback = self.pending_feedback
        self.pending_feedback = None
        return feedback

    def stop(self) -> None:
        self.active_skill = None
        self.pending_feedback = None

    def _run_skill(self, skill_call: SkillCall, observation: Any) -> ExecutionFeedback:
        roco_plan = crie_call_to_roco_skill_plan(skill_call, self.agent_names, step_id=skill_call.key)
        validation = self.validator.validate(roco_plan, observation)
        if not validation.valid:
            return self._failure_feedback(
                skill_call,
                self._failure_code_from_validation(validation),
                validation.to_feedback(),
                {"validation_issues": [issue.to_dict() for issue in validation.issues], "roco_plan": roco_plan.to_dict()},
            )

        try:
            roco_plan.prepared_execution = self.compiler.compile(roco_plan, observation)
        except Exception as exc:
            code = getattr(exc, "code", "COMPILATION_FAILED")
            return self._failure_feedback(
                skill_call,
                FailureCode.POSTCONDITION_FAILED,
                str(exc),
                {"compile_error_code": code, "roco_plan": roco_plan.to_dict()},
            )

        result = self.executor.execute(roco_plan, observation, artifact_dir=self.artifact_dir)
        final_obs = self._get_observation()
        postcondition = pack_subtask_success(self.env, final_obs, skill_call)
        if result.success and postcondition:
            return self._success_feedback(skill_call, result, final_obs)
        failure_code = self._failure_code_from_execution(result, postcondition)
        message = result.reason or "PackGrocery execution did not satisfy the subtask postcondition."
        return self._failure_feedback(
            skill_call,
            failure_code,
            message,
            {
                "execution_result": result.to_dict(),
                "postcondition_satisfied": bool(postcondition),
                "packed_slot": self._packed_slot(final_obs, skill_call),
                "roco_plan": roco_plan.to_dict(),
            },
            result=result,
        )

    def _success_feedback(self, skill_call: SkillCall, result: SkillExecutionResult, observation: Any) -> ExecutionFeedback:
        failure = FailureState(False, FailureCode.NONE)
        uncertainty_raw = self.uncertainty_reporter.report(result, True, failure)
        return ExecutionFeedback(
            skill_call=skill_call,
            status=BTStatus.SUCCESS,
            progress=ProgressState(
                stage=ProgressStage.STABLE_SUCCESS,
                score=1.0,
                elapsed_steps=int(result.num_sim_steps),
                postcondition_satisfied=True,
                evidence={"execution_result": result.to_dict(), "packed_slot": self._packed_slot(observation, skill_call)},
            ),
            uncertainty=self._uncertainty_state(uncertainty_raw),
            failure=failure,
            message="PackGrocery subtask succeeded.",
            raw_info=dict(uncertainty_raw, execution_result=result.to_dict(), packed_slot=self._packed_slot(observation, skill_call)),
        )

    def _failure_feedback(
        self,
        skill_call: SkillCall,
        code: FailureCode,
        message: str,
        evidence: Mapping[str, Any],
        result: Optional[SkillExecutionResult] = None,
    ) -> ExecutionFeedback:
        failure = FailureState(True, code, "error", message, evidence)
        uncertainty_raw = self.uncertainty_reporter.report(result, False, failure)
        raw_info = dict(evidence)
        raw_info.update(uncertainty_raw)
        return ExecutionFeedback(
            skill_call=skill_call,
            status=BTStatus.FAILURE,
            progress=ProgressState(
                stage=ProgressStage.FAILED,
                score=0.0,
                elapsed_steps=int(getattr(result, "num_sim_steps", 0) if result is not None else 0),
                postcondition_satisfied=False,
                evidence=evidence,
            ),
            uncertainty=self._uncertainty_state(uncertainty_raw),
            failure=failure,
            message=message,
            raw_info=raw_info,
        )

    def _uncertainty_state(self, raw: Mapping[str, Any]) -> UncertaintyState:
        confidence = float(raw.get("confidence", 0.5))
        entropy = float(raw.get("entropy", 1.0 - confidence))
        if entropy >= 0.66:
            risk = "high"
        elif entropy >= 0.33:
            risk = "medium"
        else:
            risk = "low"
        return UncertaintyState(
            confidence=confidence,
            uncertainty=entropy,
            risk_level=risk,
            source="fake_pack_grocery_policy_metadata",
            evidence=dict(raw),
        )

    def _get_observation(self):
        if hasattr(self.env, "get_obs"):
            return self.env.get_obs()
        if hasattr(self.env, "get_observation"):
            return self.env.get_observation()
        return None

    def _packed_slot(self, observation: Any, skill_call: SkillCall) -> Optional[str]:
        object_name = skill_call.arguments.get("object")
        if object_name and hasattr(self.env, "get_packed_slot_for_object"):
            try:
                return self.env.get_packed_slot_for_object(observation, object_name)
            except Exception:
                return None
        return None

    def _failure_code_from_validation(self, validation: Any) -> FailureCode:
        codes = set(issue.code for issue in validation.issues)
        if "TARGET_OCCUPIED" in codes:
            return FailureCode.TARGET_OCCUPIED
        if "UNKNOWN_OBJECT" in codes:
            return FailureCode.WRONG_OBJECT
        if "UNKNOWN_TARGET" in codes:
            return FailureCode.WRONG_TARGET
        if "NO_PROGRESS" in codes:
            return FailureCode.NO_PROGRESS
        return FailureCode.POSTCONDITION_FAILED

    def _failure_code_from_execution(self, result: SkillExecutionResult, postcondition: bool) -> FailureCode:
        if result.status == SkillExecutionStatus.TIMEOUT:
            return FailureCode.TIMEOUT
        if result.status == SkillExecutionStatus.MOTION_PLANNING_FAILED:
            return FailureCode.NO_PROGRESS
        if result.status in (SkillExecutionStatus.INVALID_PLAN, SkillExecutionStatus.NOT_PREPARED):
            return FailureCode.POSTCONDITION_FAILED
        if result.status == SkillExecutionStatus.INTERRUPTED:
            return FailureCode.SAFETY_CONFLICT
        if result.success and not postcondition:
            return FailureCode.POSTCONDITION_FAILED
        return FailureCode.UNKNOWN


def build_pack_registry(env: Any) -> SkillRegistry:
    from rocobench.skills.pack_grocery import build_pack_grocery_skill_registry

    return build_pack_grocery_skill_registry(list(getattr(env, "robot_name_map", {}).values()))


def parse_object_targets(value: str) -> Dict[str, str]:
    if not value:
        return {}
    parsed = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Object target entries must use object:container syntax.")
        object_name, container = item.split(":", 1)
        parsed[object_name.strip()] = container.strip()
    return parsed


def supported_uncertainty_profiles() -> Tuple[str, ...]:
    return ("nominal", "medium", "low_confidence")
