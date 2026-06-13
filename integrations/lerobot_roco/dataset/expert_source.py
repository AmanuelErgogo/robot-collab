"""RRT expert fixture construction for Phase 2 collection."""

from typing import Any, Iterable, List, Mapping, Optional

from .config import DatasetCollectionConfig
from .recorder import TransitionObserver


def build_skill_plan(
    active_agent: str,
    object_name: str,
    target_name: str,
    agent_names: Iterable[str] = ("Alice", "Bob"),
) -> Any:
    from rocobench.skills.models import SkillCall, SkillPlan

    calls: List[Any] = []
    for agent_name in agent_names:
        if agent_name == active_agent:
            raw = "PUT_OBJECT_IN_CONTAINER(object={}, container={})".format(object_name, target_name)
            calls.append(
                SkillCall(
                    agent_name=agent_name,
                    skill_name="PUT_OBJECT_IN_CONTAINER",
                    arguments={"object": object_name, "container": target_name},
                    raw_action=raw,
                )
            )
        else:
            calls.append(
                SkillCall(
                    agent_name=agent_name,
                    skill_name="WAIT",
                    arguments={},
                    raw_action="WAIT()",
                )
            )
    return SkillPlan(calls=calls, parsed_proposal="EXECUTE")


def canonical_skill_call(plan: Any) -> str:
    return plan.get_action_desp()


def natural_language_instruction(object_name: str, target_name: str, agent_name: str) -> str:
    return "{} put the {} into {}.".format(agent_name, object_name, target_name)


def prepare_rrt_plan(env: Any, obs: Any, plan: Any) -> Any:
    from prompting.parser import LLMResponseParser
    from rocobench.skills.compiler import RRTSkillCompiler

    legacy_parser = LLMResponseParser(env, "action_only", env.robot_name_map, ["NAME", "ACTION"])
    compiler = RRTSkillCompiler(env, legacy_parser)
    plan.prepared_execution = compiler.compile(plan, obs)
    return plan


def _legacy_stage_response(active_agent: str, passive_agents: Iterable[str], active_action: str) -> str:
    lines = ["EXECUTE", "NAME {} ACTION {}".format(active_agent, active_action)]
    for agent_name in passive_agents:
        lines.append("NAME {} ACTION WAIT".format(agent_name))
    return "\n".join(lines)


def _compile_legacy_response(env: Any, obs: Any, response: str) -> Any:
    from prompting.parser import LLMResponseParser

    legacy_parser = LLMResponseParser(env, "action_only", env.robot_name_map, ["NAME", "ACTION"])
    success, message, compiled_plans = legacy_parser.parse(obs, response)
    if not success:
        raise RuntimeError(message)
    return list(compiled_plans)


def build_rrt_executor(
    env: Any,
    transition_observer: Optional[TransitionObserver] = None,
    max_sim_steps: int = 5000,
) -> Any:
    from rocobench.skills.executor import RRTSkillExecutor

    return RRTSkillExecutor(
        env,
        env.get_sim_robots() if hasattr(env, "get_sim_robots") else env.robots,
        transition_observer=transition_observer,
        max_sim_steps=max_sim_steps,
    )


def execute_staged_put_object_in_container(
    env: Any,
    obs: Any,
    plan: Any,
    config: DatasetCollectionConfig,
    transition_observer: Optional[TransitionObserver] = None,
) -> Any:
    """Execute Phase 2's PUT skill as PICK then PLACE with refreshed state.

    The normal Phase 1 compiler keeps legacy behavior unchanged. Dataset
    collection needs the PLACE stage to be compiled after PICK so the legacy
    parser can see the object in the gripper and set in-hand planning metadata.
    """
    from rocobench.skills.models import PreparedSkillExecution, SkillExecutionResult, SkillExecutionStatus, SkillPlan

    active_call = None
    for call in plan.calls:
        if call.agent_name == config.active_agent and call.skill_name == config.skill_id:
            active_call = call
            break
    if active_call is None:
        return SkillExecutionResult(
            success=False,
            status=SkillExecutionStatus.INVALID_PLAN,
            reason="No active {} call found.".format(config.skill_id),
            num_sim_steps=0,
            reward=0.0,
            done=False,
            info={},
        )

    object_name = active_call.arguments["object"]
    target_name = active_call.arguments["container"]
    passive_agents = tuple(agent for agent in config.passive_agents if agent != config.active_agent)
    stages = (
        ("pick", _legacy_stage_response(config.active_agent, passive_agents, "PICK {}".format(object_name))),
        ("place", _legacy_stage_response(config.active_agent, passive_agents, "PLACE {} {}".format(object_name, target_name))),
    )

    total_steps = 0
    reward = 0.0
    done = False
    info = {}
    stage_results = []
    current_obs = obs
    executor = build_rrt_executor(
        env,
        transition_observer=transition_observer,
        max_sim_steps=config.max_episode_steps,
    )
    for stage_name, response in stages:
        try:
            compiled_plans = _compile_legacy_response(env, current_obs, response)
        except Exception as exc:
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.INVALID_PLAN,
                reason="{} stage compilation failed: {}".format(stage_name, exc),
                num_sim_steps=total_steps,
                reward=reward,
                done=done,
                info=info,
                metadata={"stage_results": stage_results, "failed_stage": stage_name},
            )
        stage_plan = SkillPlan(
            calls=list(plan.calls),
            parsed_proposal=response,
            plan_id="{}_{}".format(plan.plan_id, stage_name),
        )
        stage_plan.prepared_execution = PreparedSkillExecution(
            backend_name="rrt",
            source_plan_id=plan.plan_id,
            compiled_plans=compiled_plans,
            metadata={"synthetic_response": response, "stage": stage_name},
        )
        result = executor.execute(stage_plan, current_obs)
        total_steps += int(result.num_sim_steps)
        reward = float(result.reward)
        done = bool(result.done)
        info = dict(result.info)
        stage_results.append({"stage": stage_name, "result": result.to_dict()})
        if not result.success:
            return SkillExecutionResult(
                success=False,
                status=result.status,
                reason="{} stage failed: {}".format(stage_name, result.reason),
                num_sim_steps=total_steps,
                reward=reward,
                done=done,
                info=info,
                metadata={"stage_results": stage_results, "failed_stage": stage_name},
            )
        current_obs = env.get_obs() if hasattr(env, "get_obs") else current_obs

    return SkillExecutionResult(
        success=True,
        status=SkillExecutionStatus.SUCCESS,
        reason="",
        num_sim_steps=total_steps,
        reward=reward,
        done=done,
        info=info,
        metadata={"stage_results": stage_results},
    )


def check_put_object_postcondition(env: Any, obs: Any, object_name: str, target_name: str) -> bool:
    if hasattr(env, "get_packed_slot_for_object"):
        try:
            return env.get_packed_slot_for_object(obs, object_name) == target_name
        except Exception:
            return False
    if hasattr(env, "get_reward_done"):
        try:
            reward, done = env.get_reward_done(obs)
            return bool(done or float(reward) > 0)
        except Exception:
            return False
    return False


def build_episode_metadata(
    config: DatasetCollectionConfig,
    episode_id: str,
    variation: Any,
    plan: Any,
    result: Any,
    postcondition_success: bool,
    robot_name: Optional[str],
) -> Mapping[str, Any]:
    success = bool(getattr(result, "success", False) and postcondition_success)
    termination_reason = "success" if success else getattr(getattr(result, "status", None), "value", "failed")
    return {
        "episode_id": episode_id,
        "seed": int(variation.seed),
        "variation_id": variation.variation_id,
        "task_id": config.task_id,
        "skill_id": config.skill_id,
        "canonical_skill_call": canonical_skill_call(plan),
        "natural_language_instruction": natural_language_instruction(
            variation.object_name,
            variation.target_name,
            variation.agent_name,
        ),
        "agent_name": variation.agent_name,
        "robot_name": robot_name or variation.agent_name,
        "object_name": variation.object_name,
        "target_name": variation.target_name,
        "success": success,
        "termination_reason": termination_reason,
        "frame_count": None,
        "expert_backend": config.expert_backend,
        "expert_plan_id": plan.plan_id,
        "executor_success": bool(getattr(result, "success", False)),
        "postcondition_success": bool(postcondition_success),
        "result": result.to_dict() if hasattr(result, "to_dict") else str(result),
    }
