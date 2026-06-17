"""Baseline runners for RoCoBench-Pack-Skills-v1."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .config import EvaluationRunConfig, METHOD_ACT, METHOD_HOLD, METHOD_LEARNED_RRT_FALLBACK, METHOD_RANDOM, METHOD_RRT
from .env import make_env
from .metrics import percentile
from .tasks import BenchmarkTask
from .variations import BenchmarkVariation


class BaselineError(RuntimeError):
    """Raised when a baseline fails before producing a valid episode result."""


class BaselineUnsupported(BaselineError):
    """Raised when a baseline/task combination is intentionally unsupported."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: str, data: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(data)), f, indent=2, sort_keys=True)


def _env_fps(env: Any) -> float:
    metadata = getattr(env, "metadata", {}) or {}
    fps = float(metadata.get("render_fps", 0.0) or 0.0)
    return fps if fps > 0 else 1.0


def _single_active_agent(task: BenchmarkTask) -> str:
    if len(task.active_agents) != 1:
        raise BaselineUnsupported("bridge baseline supports single-agent tasks only: {}".format(task.task_id))
    return task.active_agents[0]


def _task_instruction(variation: BenchmarkVariation) -> str:
    if not variation.expected_skill_plan:
        return "Complete the assigned RoCoBench packing skill."
    item = variation.expected_skill_plan[0]
    return "Put the {} into the {}.".format(item.get("object", "assigned object"), item.get("container", "assigned slot"))


@dataclass
class BridgeStepBaseline:
    method: str
    config: EvaluationRunConfig
    env_factory: Callable[..., Any] = make_env

    def _action(self, env: Any, rng: np.random.RandomState) -> np.ndarray:
        if self.method == METHOD_HOLD:
            if hasattr(env, "hold_action"):
                return np.asarray(env.hold_action(), dtype=np.float32)
            return ((env.action_space.low + env.action_space.high) / 2.0).astype(np.float32)
        if self.method == METHOD_RANDOM:
            return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        raise BaselineUnsupported("unsupported bridge step method: {}".format(self.method))

    def run(
        self,
        task: BenchmarkTask,
        variation: BenchmarkVariation,
        episode_index: int,
        episode_dir: str,
    ) -> Dict[str, Any]:
        active_agent = _single_active_agent(task)
        cfg = {
            "endpoint": self.config.endpoint,
            "active_agent": active_agent,
            "request_timeout_ms": self.config.request_timeout_ms,
            "max_episode_steps": task.max_steps,
        }
        env = self.env_factory(cfg=cfg)
        rng = np.random.RandomState(int(variation.seed))
        os.makedirs(episode_dir, exist_ok=True)
        events_path = os.path.join(episode_dir, "events.jsonl")
        success = False
        terminated = False
        truncated = False
        termination_reason = "MAX_STEPS"
        steps = 0
        collisions = 0
        drops = 0
        final_info: Dict[str, Any] = {}
        start = time.perf_counter()
        try:
            _obs, info = env.reset(seed=int(variation.seed))
            final_info = dict(info or {})
            with open(events_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"event": "RESET", "seed": int(variation.seed), "info": _json_safe(final_info)}, sort_keys=True) + "\n")
            if bool(final_info.get("is_success", False)):
                success = True
                terminated = True
                termination_reason = "SUCCESS"
            while not terminated and not truncated and steps < int(task.max_steps):
                action = self._action(env, rng)
                next_obs, reward, step_terminated, step_truncated, info = env.step(action)
                del next_obs
                steps += 1
                final_info = dict(info or {})
                collisions += int(bool(final_info.get("collision", False) or final_info.get("contact_violation", False)))
                drops += int(bool(final_info.get("object_lost", False)))
                with open(events_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "STEP",
                                "env_step": steps,
                                "reward": float(reward),
                                "is_success": bool(final_info.get("is_success", False)),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if bool(final_info.get("is_success", False)):
                    success = True
                    terminated = True
                    termination_reason = "SUCCESS"
                elif bool(final_info.get("unsafe_state", False)):
                    terminated = True
                    termination_reason = "UNSAFE_STATE"
                elif bool(final_info.get("object_lost", False)):
                    terminated = True
                    termination_reason = "OBJECT_LOST"
                elif bool(step_terminated):
                    terminated = True
                    termination_reason = "TERMINATED"
                elif bool(step_truncated):
                    truncated = True
                    termination_reason = "MAX_STEPS"
            if not terminated and not truncated and steps >= int(task.max_steps):
                truncated = True
                termination_reason = "MAX_STEPS"
        finally:
            try:
                env.close()
            except Exception:
                pass
        elapsed = time.perf_counter() - start
        result = {
            "success": bool(success),
            "overall_success": bool(success),
            "learned_success": False,
            "fallback_success": False,
            "termination_reason": termination_reason,
            "num_env_steps": int(steps),
            "sim_time_s": float(steps) / _env_fps(env),
            "wall_time_s": elapsed,
            "collisions": collisions,
            "drops": drops,
            "artifact_dir": episode_dir,
            "extra": {"final_info": final_info, "baseline": self.method},
        }
        _write_json(os.path.join(episode_dir, "result.json"), result)
        return result


@dataclass
class ACTBaseline:
    config: EvaluationRunConfig
    env_factory: Callable[..., Any] = make_env
    _policy: Optional[Any] = None

    def run(
        self,
        task: BenchmarkTask,
        variation: BenchmarkVariation,
        episode_index: int,
        episode_dir: str,
    ) -> Dict[str, Any]:
        del episode_index
        active_agent = _single_active_agent(task)
        if not self.config.policy_path:
            raise BaselineError("--policy-path is required for ACT baseline")

        from integrations.lerobot_roco.evaluation.artifacts import EpisodeArtifactWriter
        from integrations.lerobot_roco.evaluation.config import Phase4EvaluationConfig
        from integrations.lerobot_roco.evaluation.policy_loader import load_lerobot_policy
        from integrations.lerobot_roco.evaluation.rollout import run_policy_rollout

        phase4_config = Phase4EvaluationConfig(
            name="phase8_{}_{}".format(task.task_id.replace(".", "_"), variation.variation_id.replace("-", "_")),
            checkpoint_dir=self.config.policy_path,
            output_root=episode_dir,
            endpoint=self.config.endpoint,
            active_agent=active_agent,
            split="test",
            frozen_suite=True,
            seeds=(int(variation.seed),),
            max_steps=int(task.max_steps),
            execution_horizon=self.config.execution_horizon,
            action_bound_tolerance=float(self.config.action_bound_tolerance),
            record_video=bool(self.config.record_video),
            render_every_steps=int(self.config.render_every_steps),
            request_timeout_ms=int(self.config.request_timeout_ms),
            task_instruction=_task_instruction(variation),
            require_lerobot=bool(self.config.require_lerobot),
            overwrite=True,
        )
        if self._policy is None:
            self._policy = load_lerobot_policy(phase4_config)
        env = self.env_factory(
            cfg={
                "endpoint": self.config.endpoint,
                "active_agent": active_agent,
                "request_timeout_ms": self.config.request_timeout_ms,
                "max_episode_steps": task.max_steps,
            }
        )
        try:
            writer = EpisodeArtifactWriter(
                root_dir=episode_dir,
                episode_name="policy_rollout",
                overwrite=True,
            )
            rollout = run_policy_rollout(env, self._policy, phase4_config, writer, seed=int(variation.seed), episode_index=0)
        finally:
            try:
                env.close()
            except Exception:
                pass

        latencies = [float(x) for x in rollout.inference_latency_ms]
        result = {
            "success": bool(rollout.success),
            "overall_success": bool(rollout.success),
            "learned_success": bool(rollout.success),
            "fallback_success": False,
            "termination_reason": rollout.termination_reason,
            "num_env_steps": int(rollout.num_env_steps),
            "sim_time_s": float(rollout.sim_time),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "action_violations": int(rollout.action_bound_violations),
            "artifact_dir": rollout.artifact_dir,
            "extra": {"rollout": rollout.to_dict(), "task_instruction": _task_instruction(variation)},
        }
        _write_json(os.path.join(episode_dir, "result.json"), result)
        return result


@dataclass
class RRTBaseline:
    config: EvaluationRunConfig

    def _response_for_variation(self, env: Any, variation: BenchmarkVariation) -> str:
        return self._response_for_items(env, variation.expected_skill_plan)

    def _response_for_items(self, env: Any, items: Any) -> str:
        agent_names = list(env.robot_name_map.values())
        by_agent = {str(item["agent_name"]): dict(item) for item in items}
        lines = ["EXECUTE"]
        for agent in agent_names:
            item = by_agent.get(agent)
            if item is None:
                lines.append("NAME {} ACTION WAIT()".format(agent))
            else:
                lines.append(
                    "NAME {} ACTION PUT_OBJECT_IN_CONTAINER(object={}, container={})".format(
                        agent,
                        item["object"],
                        item["container"],
                    )
                )
        return "\n".join(lines)

    def _postcondition(self, env: Any, variation: BenchmarkVariation) -> bool:
        obs = env.get_obs()
        for item in variation.expected_skill_plan:
            object_name = str(item["object"])
            target = str(item["container"])
            if env.get_packed_slot_for_object(obs, object_name) != target:
                return False
        return True

    def run(
        self,
        task: BenchmarkTask,
        variation: BenchmarkVariation,
        episode_index: int,
        episode_dir: str,
    ) -> Dict[str, Any]:
        del episode_index
        if task.task_id == "pack.concurrent.safe_two_agent":
            raise BaselineUnsupported("RRT baseline is sequential and does not claim Phase 7 concurrent packing.")
        os.environ.setdefault("PYTHONBREAKPOINT", "0")
        try:
            from prompting.parser import LLMResponseParser
            from prompting.skill_parser import SkillResponseParser
            from rocobench.envs import PackGroceryTask
            from rocobench.skills import PackGrocerySkillPlanValidator, RRTSkillCompiler, RRTSkillExecutor, build_pack_grocery_skill_registry
        except Exception as exc:
            raise BaselineError("RRT baseline requires Python 3.8 RoCo simulator dependencies: {}".format(exc)) from exc

        env = PackGroceryTask(
            render_cameras=["teaser"],
            image_hw=(64, 64),
            render_freq=1000,
            randomize_init=bool(variation.metadata.get("randomize_init", True)),
            render_point_cloud=False,
        )
        env.seed(np_seed=int(variation.seed))
        obs = env.reset(reload=True)
        if obs is None:
            obs = env.get_obs()
        os.makedirs(episode_dir, exist_ok=True)

        agent_names = list(env.robot_name_map.values())
        registry = build_pack_grocery_skill_registry(agent_names)
        parser = SkillResponseParser(registry, agent_names)
        response = self._response_for_variation(env, variation)
        parse_ok, parse_message, plans = parser.parse(obs, response)
        if not parse_ok or not plans:
            result = {
                "success": False,
                "overall_success": False,
                "learned_success": False,
                "fallback_success": False,
                "termination_reason": "INVALID_PLAN",
                "num_env_steps": 0,
                "sim_time_s": 0.0,
                "plan_valid": False,
                "artifact_dir": episode_dir,
                "extra": {"parse_message": parse_message, "response": response},
            }
            _write_json(os.path.join(episode_dir, "result.json"), result)
            return result

        plan = plans[0]
        validator = PackGrocerySkillPlanValidator(env, registry, agent_names)
        validation = validator.validate(plan, obs)
        if not validation.valid:
            result = {
                "success": False,
                "overall_success": False,
                "learned_success": False,
                "fallback_success": False,
                "termination_reason": "INVALID_PLAN",
                "num_env_steps": 0,
                "sim_time_s": 0.0,
                "plan_valid": False,
                "artifact_dir": episode_dir,
                "extra": {"validation_issues": [issue.to_dict() for issue in validation.issues], "response": response},
            }
            _write_json(os.path.join(episode_dir, "result.json"), result)
            return result

        execution_plan = plan
        execution_response = response
        motion_overrides = dict(variation.metadata.get("expert_motion_target_overrides", {}) or {})
        if motion_overrides:
            execution_items = []
            for item in variation.expected_skill_plan:
                copied = dict(item)
                copied["container"] = str(motion_overrides.get(str(copied["container"]), copied["container"]))
                execution_items.append(copied)
            execution_response = self._response_for_items(env, execution_items)
            parse_ok, parse_message, execution_plans = parser.parse(obs, execution_response)
            if not parse_ok or not execution_plans:
                result = {
                    "success": False,
                    "overall_success": False,
                    "learned_success": False,
                    "fallback_success": False,
                    "termination_reason": "INVALID_EXECUTION_PLAN",
                    "num_env_steps": 0,
                    "sim_time_s": 0.0,
                    "plan_valid": True,
                    "artifact_dir": episode_dir,
                    "extra": {"parse_message": parse_message, "response": response, "execution_response": execution_response},
                }
                _write_json(os.path.join(episode_dir, "result.json"), result)
                return result
            execution_plan = execution_plans[0]

        start = time.perf_counter()
        legacy_parser = LLMResponseParser(
            env,
            "action_only",
            env.robot_name_map,
            ["NAME", "ACTION"],
            use_prepick=env.use_prepick,
            use_preplace=env.use_preplace,
        )
        compiler = RRTSkillCompiler(env, legacy_parser)
        execution_plan.prepared_execution = compiler.compile(execution_plan, obs)
        executor = RRTSkillExecutor(env, env.get_sim_robots(), max_sim_steps=int(task.max_steps))
        skill_result = executor.execute(execution_plan, obs, artifact_dir=os.path.join(episode_dir, "rrt"))
        wall_time_s = time.perf_counter() - start
        postcondition = self._postcondition(env, variation)
        success = bool(skill_result.success and postcondition)
        if success:
            termination_reason = "SUCCESS"
        elif skill_result.success and not postcondition:
            termination_reason = "POSTCONDITION_FAILED"
        else:
            termination_reason = skill_result.status.value.upper()
        result = {
            "success": success,
            "overall_success": success,
            "learned_success": False,
            "fallback_success": False,
            "termination_reason": termination_reason,
            "num_env_steps": int(skill_result.num_sim_steps),
            "sim_time_s": float(skill_result.num_sim_steps) / 50.0,
            "plan_valid": True,
            "artifact_dir": episode_dir,
            "wall_time_s": wall_time_s,
            "extra": {
                "response": response,
                "execution_response": execution_response,
                "expert_motion_target_overrides": motion_overrides,
                "skill_result": skill_result.to_dict(),
                "postcondition_met": postcondition,
            },
        }
        _write_json(os.path.join(episode_dir, "result.json"), result)
        return result


@dataclass
class LearnedRRTFallbackBaseline:
    config: EvaluationRunConfig

    def run(self, task: BenchmarkTask, variation: BenchmarkVariation, episode_index: int, episode_dir: str) -> Dict[str, Any]:
        del task, variation, episode_index, episode_dir
        raise BaselineUnsupported(
            "learned_rrt_fallback requires an injected Phase 5 LearnedSkillExecutor and RRT fallback controller; "
            "the CLI does not fabricate this path."
        )


def build_baseline(config: EvaluationRunConfig) -> Any:
    if config.method in (METHOD_HOLD, METHOD_RANDOM):
        return BridgeStepBaseline(config.method, config)
    if config.method == METHOD_ACT:
        return ACTBaseline(config)
    if config.method == METHOD_RRT:
        return RRTBaseline(config)
    if config.method == METHOD_LEARNED_RRT_FALLBACK:
        return LearnedRRTFallbackBaseline(config)
    raise BaselineUnsupported("unsupported method: {}".format(config.method))
