"""Shared construction of the real RoCo LLM planner and RRT executor.

Extracted from ``scripts/run_crie_bt_sim.py`` so that **both** the legacy runner
and the new pipeline (`rocobench.crie_bt.pipeline.roco_backend`) build the real
Gemini/OpenAI prompter planner and the RRT executor from one place, instead of
duplicating the wiring.  Heavy imports (MuJoCo, prompting, llm_api) are lazy so
importing this module stays cheap.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional


def _usage_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {str(k): v for k, v in usage.items()}
    try:
        return {str(k): v for k, v in dict(usage).items()}
    except Exception:
        return {}


def instrument_prompter_llm_usage(prompter: Any) -> None:
    """Record LLM latency and token usage at the prompter boundary."""
    prompter._crie_bt_llm_call_latencies_s = []
    prompter._crie_bt_llm_usage = []
    original_query_once = prompter.query_once

    def query_once(*call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            response, usage = original_query_once(*call_args, **call_kwargs)
        except Exception:
            prompter._crie_bt_llm_call_latencies_s.append(round(time.perf_counter() - started, 6))
            raise
        prompter._crie_bt_llm_call_latencies_s.append(round(time.perf_counter() - started, 6))
        prompter._crie_bt_llm_usage.append(_usage_dict(usage))
        return response, usage

    prompter.query_once = query_once


def uses_shared_llm_api(llm_source: str) -> bool:
    return str(llm_source).strip().lower().startswith("gemini")


def prompter_llm_source(llm_source: str) -> str:
    # The prompting package validates known source names; route Gemini through
    # the shared llm_api bridge while presenting a name it accepts.
    return "gpt-4" if uses_shared_llm_api(llm_source) else llm_source


def install_llm_api_query_bridge(
    prompter: Any,
    llm_source: str,
    api_key_path: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> None:
    """Route ``prompter.query_once`` through ``llm_api`` (used for Gemini/Vertex)."""
    from llm_api import create_llm_client

    client = create_llm_client(llm_source, api_key_path=(api_key_path or None))

    def query_once(system_prompt, user_prompt="", max_query=None):
        del max_query
        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        response = client.generate(messages=messages, max_tokens=int(max_tokens), temperature=float(temperature))
        return response.text, response.usage

    prompter.query_once = query_once


def build_legacy_prompt_planner(
    env: Any,
    task_id: str,
    planner_mode: str,
    llm_source: str = "gpt-4",
    api_key_path: str = "",
    save_dir: Optional[str] = None,
    llm_output_mode: str = "action_only",
    direct_waypoints: int = 0,
    max_failed_waypoints: int = 1,
    max_tokens: int = 1024,
    num_replans: int = 3,
    temperature: float = 0.0,
    max_calls_per_round: int = 10,
    use_history: bool = True,
    use_feedback: bool = True,
    plan_horizon: int = 1,
) -> Any:
    """Build a real-LLM ``LegacyPromptPlanner`` (SingleThread for chat/plan, Dialog for dialog)."""
    from prompting import DialogPrompter, FeedbackManager, LLMResponseParser, SingleThreadPrompter
    from rocobench import MultiArmRRT
    from rocobench.crie_bt.legacy_tasks import LegacyPromptPlanner, agent_names_for_env

    response_keywords = ["NAME", "ACTION"]
    if llm_output_mode == "action_and_path":
        response_keywords.append("PATH")
    parser = LLMResponseParser(
        env,
        llm_output_mode,
        env.robot_name_map,
        response_keywords,
        int(direct_waypoints),
        use_prepick=getattr(env, "use_prepick", False),
        use_preplace=getattr(env, "use_preplace", False),
        split_parsed_plans=False,
    )
    rrt_planner = MultiArmRRT(
        env.physics,
        robots=env.get_sim_robots(),
        graspable_object_names=env.get_graspable_objects(),
        allowed_collision_pairs=env.get_allowed_collision_pairs(),
    )
    feedback_manager = FeedbackManager(
        env=env,
        planner=rrt_planner,
        llm_output_mode=llm_output_mode,
        robot_name_map=env.robot_name_map,
        step_std_threshold=getattr(env, "waypoint_std_threshold", 0.1),
        max_failed_waypoints=int(max_failed_waypoints),
    )
    if planner_mode in ("plan", "chat"):
        prompter = SingleThreadPrompter(
            env=env,
            parser=parser,
            feedback_manager=feedback_manager,
            max_tokens=int(max_tokens),
            debug_mode=False,
            use_waypoints=(llm_output_mode == "action_and_path"),
            use_history=use_history,
            num_replans=int(num_replans),
            comm_mode=planner_mode,
            temperature=float(temperature),
            llm_source=prompter_llm_source(llm_source),
        )
    else:
        prompter = DialogPrompter(
            env=env,
            parser=parser,
            feedback_manager=feedback_manager,
            max_tokens=int(max_tokens),
            debug_mode=False,
            robot_name_map=env.robot_name_map,
            max_calls_per_round=int(max_calls_per_round),
            use_waypoints=(llm_output_mode == "action_and_path"),
            use_history=use_history,
            use_feedback=use_feedback,
            num_replans=int(num_replans),
            temperature=float(temperature),
            llm_source=prompter_llm_source(llm_source),
        )
    if uses_shared_llm_api(llm_source):
        install_llm_api_query_bridge(prompter, llm_source, api_key_path=api_key_path,
                                     max_tokens=max_tokens, temperature=temperature)
    instrument_prompter_llm_usage(prompter)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    return LegacyPromptPlanner(
        task_id=task_id,
        prompter=prompter,
        planner_mode=planner_mode,
        agent_names=agent_names_for_env(env),
        save_dir=save_dir,
        plan_horizon=plan_horizon,
    )


def build_legacy_rrt_executor(
    env: Any,
    task_id: str,
    max_sim_steps: int = 5000,
    artifact_dir: Optional[str] = None,
    uncertainty_profile: str = "nominal",
) -> Any:
    """Build the RRT executor adapter that runs EXECUTE blocks through MuJoCo."""
    from prompting.parser import LLMResponseParser
    from rocobench.crie_bt.legacy_tasks import FakeLegacyUncertaintyReporter, LegacyTaskRRTExecutorAdapter
    from rocobench.skills import RRTSkillExecutor

    legacy_parser = LLMResponseParser(
        env,
        "action_only",
        env.robot_name_map,
        ["NAME", "ACTION"],
        use_prepick=getattr(env, "use_prepick", False),
        use_preplace=getattr(env, "use_preplace", False),
    )
    legacy_executor = RRTSkillExecutor(env=env, robots=env.get_sim_robots(), max_sim_steps=int(max_sim_steps))
    return LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id=task_id,
        parser=legacy_parser,
        executor=legacy_executor,
        artifact_dir=artifact_dir,
        uncertainty_reporter=FakeLegacyUncertaintyReporter(uncertainty_profile),
    )


def llm_usage_from_planner(planner: Any) -> Dict[str, List[Any]]:
    """Collect LLM latency + usage recorded by ``instrument_prompter_llm_usage``."""
    prompter = getattr(planner, "prompter", None)
    if prompter is None:
        return {"latencies_s": [], "usage": []}
    return {
        "latencies_s": list(getattr(prompter, "_crie_bt_llm_call_latencies_s", []) or []),
        "usage": list(getattr(prompter, "_crie_bt_llm_usage", []) or []),
    }


__all__ = [
    "build_legacy_prompt_planner",
    "build_legacy_rrt_executor",
    "install_llm_api_query_bridge",
    "instrument_prompter_llm_usage",
    "llm_usage_from_planner",
    "prompter_llm_source",
    "uses_shared_llm_api",
]
