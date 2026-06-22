"""CRIE-BT adapters for legacy RoCoBench task action plans.

The non-PackGrocery tasks in this repository already expose action grammars and
the generic ``LLMResponseParser``/RRT execution path, but they do not have
typed skill registries like PackGrocery. This module wraps those existing
contracts without inventing new simulator skills.
"""

import importlib
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rocobench.skills.models import (
    PreparedSkillExecution,
    SkillCall as RoCoSkillCall,
    SkillExecutionResult,
    SkillExecutionStatus,
    SkillPlan as RoCoSkillPlan,
)

from .executor import BaseSkillExecutor
from .planner import BasePlanner
from .status import BTStatus, FailureCode, ProgressStage
from .types import CollaborativePlan, ExecutionContext, ExecutionFeedback, FailureState, PlanStep, ProgressState, SkillCall, UncertaintyState


LEGACY_ACTION_PLAN = "LEGACY_ACTION_PLAN"
WAIT = "WAIT"


TASK_CLASS_PATHS = {
    "pack": ("rocobench.envs", "PackGroceryTask"),
    "sort": ("rocobench.envs", "SortOneBlockTask"),
    "sweep": ("rocobench.envs", "SweepTask"),
    "sandwich": ("rocobench.envs", "MakeSandwichTask"),
    "rope": ("rocobench.envs", "MoveRopeTask"),
    "cabinet": ("rocobench.envs", "CabinetTask"),
}


def available_legacy_task_ids() -> Tuple[str, ...]:
    return tuple(sorted(TASK_CLASS_PATHS))


def load_task_class(task_id: str):
    task_id = normalize_task_id(task_id)
    module_name, class_name = TASK_CLASS_PATHS[task_id]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def normalize_task_id(task_id: str) -> str:
    normalized = str(task_id or "").strip().lower()
    aliases = {
        "pack_grocery": "pack",
        "grocery": "pack",
        "sorting": "sort",
        "sweeping": "sweep",
        "make_sandwich": "sandwich",
        "move_rope": "rope",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TASK_CLASS_PATHS:
        raise ValueError("Unknown RoCoBench task id: {}".format(task_id))
    return normalized


def make_legacy_task_env(
    task_id: str,
    seed: int = 0,
    render_cameras: Optional[Sequence[str]] = None,
    randomize_init: bool = False,
    render_point_cloud: bool = False,
):
    cls = load_task_class(task_id)
    kwargs = {
        "render_cameras": list(render_cameras or ["teaser"]),
        "randomize_init": bool(randomize_init),
        "render_point_cloud": bool(render_point_cloud),
    }
    env = cls(**kwargs)
    if hasattr(env, "seed"):
        env.seed(np_seed=int(seed))
    try:
        env.reset(reload=True)
    except TypeError:
        env.reset()
    return env


def agent_names_for_env(env: Any) -> List[str]:
    return list(getattr(env, "robot_name_map", {}).values())


def make_wait_response(agent_names: Sequence[str]) -> str:
    lines = ["EXECUTE"]
    for agent_name in agent_names:
        lines.append("NAME {} ACTION WAIT".format(agent_name))
    return "\n".join(lines)


def split_legacy_responses(text: str) -> List[str]:
    """Split one or more raw RoCoBench EXECUTE blocks."""

    text = str(text or "").replace("\\n", "\n").strip()
    if not text:
        return []
    responses = []  # type: List[str]
    current = []  # type: List[str]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "EXECUTE" and current:
            responses.append("\n".join(current).strip())
            current = [stripped]
            continue
        if current or stripped:
            current.append(line.rstrip())
    if current:
        responses.append("\n".join(current).strip())
    return [response for response in responses if response]


def _safe_env_text(env: Any, method_name: str) -> str:
    method = getattr(env, method_name, None)
    if method is None:
        return ""
    try:
        return str(method())
    except Exception as exc:
        return "{} unavailable: {}".format(method_name, exc)


def legacy_task_spec(env: Any, task_id: str) -> Dict[str, Any]:
    task_id = normalize_task_id(task_id)
    action_prompt = _safe_env_text(env, "get_action_prompt")
    task_context = _safe_env_text(env, "describe_task_context")
    return {
        "task": task_id,
        "source": "{}.{}".format(*TASK_CLASS_PATHS[task_id]),
        "adapter": "legacy_action_plan",
        "agents": agent_names_for_env(env),
        "skills": {
            LEGACY_ACTION_PLAN: {
                "arguments": ["response"],
                "description": "Existing RoCoBench EXECUTE/NAME/ACTION plan for this task.",
                "supported_agents": agent_names_for_env(env),
                "resources": ["all_task_agents", "task_objects"],
                "subtask_success": "legacy parser succeeds and RRT executor succeeds",
                "task_success": "env.get_reward_done(final_obs)[1]",
                "failure_checks": [
                    "LLMResponseParser rejects the response",
                    "task-specific get_task_feedback reports a problem",
                    "RRTSkillExecutor reports motion planning failure",
                    "RRTSkillExecutor reports timeout or execution failure",
                ],
            }
        },
        "actions": {
            "source": "env.get_action_prompt()",
            "prompt": action_prompt,
        },
        "observation": {
            "type": "rocobench.envs.base_env.EnvState",
            "scene_summary": "env.describe_obs(obs)",
            "task_reward_done": "env.get_reward_done(obs)",
        },
        "task_context": task_context,
        "uncertainty": {
            "source": "fake_legacy_policy_metadata",
            "fields": ["confidence", "entropy", "action_norm", "chunk_disagreement"],
            "note": "Deterministic placeholder metadata for later learned-model entropy or ensemble estimators.",
        },
    }


class LegacyActionPlanner(BasePlanner):
    """Planner that emits existing RoCoBench raw action responses."""

    def __init__(
        self,
        task_id: str,
        response: Optional[str] = None,
        agent_names: Optional[Sequence[str]] = None,
        responses: Optional[Sequence[str]] = None,
    ) -> None:
        self.task_id = normalize_task_id(task_id)
        self.responses = list(responses or split_legacy_responses(response or ""))
        self.agent_names = list(agent_names or ())
        self.calls = 0

    def generate_plan(
        self,
        task_goal: str,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        context: Optional[ExecutionContext] = None,
    ) -> CollaborativePlan:
        del observation, feedback, context
        self.calls += 1
        responses = list(self.responses or [make_wait_response(self.agent_names)])
        steps = []
        for index, response in enumerate(responses, start=1):
            skill_call = SkillCall(
                agent="ALL",
                skill_name=LEGACY_ACTION_PLAN,
                arguments={"task": self.task_id, "response": response},
                instruction="Execute legacy {} action plan {}.".format(self.task_id, index),
            )
            steps.append(
                PlanStep(
                    step_id="{}_legacy_step_{:03d}_{:03d}".format(self.task_id, self.calls, index),
                    skill_call=skill_call,
                    role_assignment={agent_name: "legacy_action_agent" for agent_name in self.agent_names},
                    explanation="Execute existing RoCoBench action plan {}.".format(index),
                    metadata={"response": response, "task": self.task_id, "response_index": index},
                )
            )
        return CollaborativePlan(
            steps=steps,
            plan_id="{}_legacy_plan_{:03d}".format(self.task_id, self.calls),
            task_goal=task_goal,
            metadata={"task": self.task_id, "planner": "legacy_action", "responses": responses},
        )


def _extract_execute_response(raw_text: str) -> str:
    raw_text = str(raw_text or "").strip()
    if "EXECUTE" not in raw_text:
        return ""
    return "EXECUTE\n" + raw_text.split("EXECUTE", 1)[1].strip()


def legacy_response_from_prompt_outputs(llm_plans: Sequence[Any], raw_outputs: Sequence[Any]) -> str:
    """Recover a raw EXECUTE block from RoCoBench prompter outputs."""

    for raw_output in reversed(list(raw_outputs or [])):
        response = _extract_execute_response(str(raw_output))
        if response:
            return response
    if llm_plans:
        parsed_proposal = str(getattr(llm_plans[0], "parsed_proposal", "") or "").strip()
        if parsed_proposal:
            if parsed_proposal.startswith("EXECUTE"):
                return parsed_proposal
            return "EXECUTE\n" + parsed_proposal
    return ""


def _augment_obs_with_feedback(observation: Any, feedback: Optional[Any]) -> Any:
    """Return observation augmented with failure context for replanning.

    The existing RoCoBench prompters accept an observation object (usually an
    EnvState) and do not have a dedicated feedback parameter. When replanning
    after a failure, we attach failure metadata as an ``_crie_bt_feedback``
    attribute so prompters that introspect observation metadata can surface it.
    Prompters that ignore unknown attributes are unaffected.
    """
    if feedback is None:
        return observation
    try:
        failure = feedback.failure
        if not failure.is_failure:
            return observation
        import copy
        augmented = copy.copy(observation)
        augmented._crie_bt_feedback = {
            "failure_code": failure.failure_code.value,
            "message": str(failure.message or ""),
            "severity": str(failure.severity or ""),
        }
        return augmented
    except Exception:
        return observation


class LegacyPromptPlanner(BasePlanner):
    """Planner adapter around existing RoCoBench plan/chat/dialog prompters."""

    def __init__(
        self,
        task_id: str,
        prompter: Any,
        planner_mode: str,
        agent_names: Sequence[str],
        save_dir: Optional[str] = None,
    ) -> None:
        if planner_mode not in ("plan", "chat", "dialog"):
            raise ValueError("planner_mode must be one of plan, chat, dialog.")
        self.task_id = normalize_task_id(task_id)
        self.prompter = prompter
        self.planner_mode = planner_mode
        self.agent_names = list(agent_names)
        self.save_dir = save_dir
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
        save_path = ""
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "planner_call_{:03d}".format(self.calls))
            os.makedirs(save_path, exist_ok=True)
        # Augment observation with failure context for replanning rounds.
        obs_for_prompter = _augment_obs_with_feedback(observation, feedback)
        ready, llm_plans, plan_feedbacks, raw_outputs = self.prompter.prompt_one_round(obs_for_prompter, save_path=save_path)
        if not ready:
            raise RuntimeError("{} planner did not produce an executable RoCoBench plan.".format(self.planner_mode))
        response = legacy_response_from_prompt_outputs(llm_plans, raw_outputs)
        if not response:
            raise RuntimeError("{} planner output did not contain an EXECUTE block.".format(self.planner_mode))
        skill_call = SkillCall(
            agent="ALL",
            skill_name=LEGACY_ACTION_PLAN,
            arguments={"task": self.task_id, "response": response},
            instruction="Execute {} planner output for {}.".format(self.planner_mode, self.task_id),
        )
        return CollaborativePlan(
            steps=[
                PlanStep(
                    step_id="{}_{}_planner_step_{:03d}".format(self.task_id, self.planner_mode, self.calls),
                    skill_call=skill_call,
                    role_assignment={agent_name: "legacy_action_agent" for agent_name in self.agent_names},
                    explanation="Execute one {} planner-generated RoCoBench action plan.".format(self.planner_mode),
                    metadata={
                        "response": response,
                        "task": self.task_id,
                        "planner_mode": self.planner_mode,
                        "plan_feedbacks": list(plan_feedbacks or []),
                        "save_path": save_path,
                    },
                )
            ],
            plan_id="{}_{}_planner_plan_{:03d}".format(self.task_id, self.planner_mode, self.calls),
            task_goal=task_goal,
            metadata={
                "task": self.task_id,
                "planner": self.planner_mode,
                "response": response,
                "plan_feedbacks": list(plan_feedbacks or []),
                "save_path": save_path,
            },
        )


class FakeLegacyUncertaintyReporter(object):
    """Deterministic placeholder for learned legacy-policy uncertainty."""

    def __init__(self, profile: str = "nominal") -> None:
        self.profile = str(profile or "nominal")

    def report(self, result: Optional[SkillExecutionResult], success: bool, failure: FailureState) -> Dict[str, Any]:
        if self.profile == "low_confidence":
            confidence = 0.42
        elif self.profile == "medium":
            confidence = 0.58
        elif failure.is_failure or not success:
            confidence = 0.24
        else:
            confidence = 0.84
        num_steps = int(getattr(result, "num_sim_steps", 0) if result is not None else 0)
        entropy = max(0.0, min(1.0, 1.0 - confidence))
        return {
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "action_norm": round(float(num_steps) / 100.0, 4),
            "chunk_disagreement": round(entropy * 0.5, 4),
            "uncertainty_source": "fake_legacy_policy_metadata",
            "fake_uncertainty": True,
        }


class LegacyTaskRRTExecutorAdapter(BaseSkillExecutor):
    """Execute an existing RoCoBench raw action plan through RRT."""

    def __init__(
        self,
        env: Any,
        task_id: str,
        parser: Any,
        executor: Any,
        artifact_dir: Optional[str] = None,
        uncertainty_reporter: Optional[FakeLegacyUncertaintyReporter] = None,
    ) -> None:
        self.env = env
        self.task_id = normalize_task_id(task_id)
        self.parser = parser
        self.executor = executor
        self.artifact_dir = artifact_dir
        self.uncertainty_reporter = uncertainty_reporter or FakeLegacyUncertaintyReporter()
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
        self.pending_feedback = self._run_legacy_response(skill_call, observation)

    def step(self, observation: Any) -> ExecutionFeedback:
        del observation
        if self.pending_feedback is None:
            if self.active_skill is None:
                raise RuntimeError("LegacyTaskRRTExecutorAdapter.step called before start_skill.")
            return self._failure_feedback(self.active_skill, FailureCode.UNKNOWN, "No legacy execution feedback was prepared.", {})
        feedback = self.pending_feedback
        self.pending_feedback = None
        return feedback

    def stop(self) -> None:
        self.active_skill = None
        self.pending_feedback = None

    def _run_legacy_response(self, skill_call: SkillCall, observation: Any) -> ExecutionFeedback:
        response = skill_call.arguments.get("response", "")
        if not response:
            return self._failure_feedback(skill_call, FailureCode.POSTCONDITION_FAILED, "Missing legacy response text.", {})

        parse_success, parse_reason, path_plans = self.parser.parse(observation, response)
        path_plans = list(path_plans)
        if not parse_success:
            return self._failure_feedback(
                skill_call,
                FailureCode.POSTCONDITION_FAILED,
                parse_reason,
                {"parse_reason": parse_reason, "response": response},
            )

        task_feedback = self._task_feedback(path_plans, observation)
        if task_feedback:
            return self._failure_feedback(
                skill_call,
                FailureCode.POSTCONDITION_FAILED,
                task_feedback,
                {"task_feedback": task_feedback, "response": response},
            )

        roco_plan = self._build_roco_plan(response, path_plans)
        result = self.executor.execute(roco_plan, observation, artifact_dir=self.artifact_dir)
        final_obs = self._get_observation()
        task_done = self._task_done(final_obs)
        if result.success:
            return self._success_feedback(skill_call, result, task_done)
        code = self._failure_code_from_execution(result)
        return self._failure_feedback(
            skill_call,
            code,
            result.reason or "Legacy task RRT execution failed.",
            {"execution_result": result.to_dict(), "task_done": bool(task_done), "response": response},
            result=result,
        )

    def _build_roco_plan(self, response: str, path_plans: Iterable[Any]) -> RoCoSkillPlan:
        calls = []
        for agent_name in agent_names_for_env(self.env):
            calls.append(RoCoSkillCall(agent_name, LEGACY_ACTION_PLAN, {"task": self.task_id}, response))
        plan = RoCoSkillPlan(calls, response)
        plan.prepared_execution = PreparedSkillExecution(
            backend_name="rrt",
            source_plan_id=plan.plan_id,
            compiled_plans=list(path_plans),
            metadata={"legacy_response": response, "task": self.task_id},
        )
        return plan

    def _task_feedback(self, path_plans: Sequence[Any], observation: Any) -> str:
        if not hasattr(self.env, "get_task_feedback"):
            return ""
        messages = []
        for path_plan in path_plans:
            try:
                feedback = self.env.get_task_feedback(path_plan, {})
            except TypeError:
                feedback = self.env.get_task_feedback(path_plan)
            if feedback:
                messages.append(str(feedback))
        return "\n".join(messages)

    def _success_feedback(self, skill_call: SkillCall, result: SkillExecutionResult, task_done: bool) -> ExecutionFeedback:
        failure = FailureState(False, FailureCode.NONE)
        uncertainty_raw = self.uncertainty_reporter.report(result, True, failure)
        return ExecutionFeedback(
            skill_call=skill_call,
            status=BTStatus.SUCCESS,
            progress=ProgressState(
                stage=ProgressStage.STABLE_SUCCESS if task_done else ProgressStage.TRANSPORTING,
                score=1.0 if task_done else 0.6,
                elapsed_steps=int(result.num_sim_steps),
                postcondition_satisfied=bool(task_done),
                evidence={"execution_result": result.to_dict(), "task_done": bool(task_done)},
            ),
            uncertainty=self._uncertainty_state(uncertainty_raw),
            failure=failure,
            message="Legacy {} action plan executed.".format(self.task_id),
            raw_info=dict(uncertainty_raw, execution_result=result.to_dict(), task_done=bool(task_done)),
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
        return UncertaintyState(confidence=confidence, uncertainty=entropy, risk_level=risk, source="fake_legacy_policy_metadata", evidence=dict(raw))

    def _get_observation(self):
        if hasattr(self.env, "get_obs"):
            return self.env.get_obs()
        if hasattr(self.env, "get_observation"):
            return self.env.get_observation()
        return None

    def _task_done(self, observation: Any) -> bool:
        if observation is None or not hasattr(self.env, "get_reward_done"):
            return False
        try:
            return bool(self.env.get_reward_done(observation)[1])
        except Exception:
            return False

    def _failure_code_from_execution(self, result: SkillExecutionResult) -> FailureCode:
        if result.status == SkillExecutionStatus.TIMEOUT:
            return FailureCode.TIMEOUT
        if result.status == SkillExecutionStatus.MOTION_PLANNING_FAILED:
            return FailureCode.NO_PROGRESS
        if result.status == SkillExecutionStatus.INTERRUPTED:
            return FailureCode.SAFETY_CONFLICT
        if result.status in (SkillExecutionStatus.INVALID_PLAN, SkillExecutionStatus.NOT_PREPARED):
            return FailureCode.POSTCONDITION_FAILED
        return FailureCode.UNKNOWN


def supported_uncertainty_profiles() -> Tuple[str, ...]:
    return ("nominal", "medium", "low_confidence")
