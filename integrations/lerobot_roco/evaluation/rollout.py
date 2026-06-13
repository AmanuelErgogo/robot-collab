"""Synchronous direct policy rollout through the Phase 0 Gym/RPC bridge."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .action_queue import ACTActionQueue, ActionQueueError, QueuedAction
from .artifacts import EpisodeArtifactWriter
from .policy_loader import PolicyInferenceError, validate_native_action_chunk
from .processor_adapter import ProcessorContractError, RoCoPolicyObservationAdapter


SUCCESS = "SUCCESS"
MAX_STEPS = "MAX_STEPS"
NONFINITE_ACTION = "NONFINITE_ACTION"
ACTION_OUT_OF_BOUNDS = "ACTION_OUT_OF_BOUNDS"
POLICY_ERROR = "POLICY_ERROR"
BRIDGE_ERROR = "BRIDGE_ERROR"
COLLISION_LIMIT = "COLLISION_LIMIT"
NO_PROGRESS = "NO_PROGRESS"
OBJECT_LOST = "OBJECT_LOST"
UNSAFE_STATE = "UNSAFE_STATE"
MANUAL_INTERRUPT = "MANUAL_INTERRUPT"


@dataclass
class PolicyRolloutResult:
    success: bool
    terminated: bool
    truncated: bool
    termination_reason: str
    num_env_steps: int
    sim_time: float
    inference_latency_ms: Sequence[float]
    action_bound_violations: int
    no_progress_events: int
    final_info: Mapping[str, Any]
    artifact_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "termination_reason": self.termination_reason,
            "num_env_steps": int(self.num_env_steps),
            "sim_time": float(self.sim_time),
            "inference_latency_ms": [float(x) for x in self.inference_latency_ms],
            "action_bound_violations": int(self.action_bound_violations),
            "no_progress_events": int(self.no_progress_events),
            "final_info": dict(self.final_info),
            "artifact_dir": self.artifact_dir,
        }


@dataclass
class NoProgressMonitor:
    window: int
    patience: int
    state_epsilon: float
    action_epsilon: float
    _states: List[np.ndarray] = field(default_factory=list)
    _actions: List[np.ndarray] = field(default_factory=list)
    _stalled_windows: int = 0

    def reset(self) -> None:
        self._states = []
        self._actions = []
        self._stalled_windows = 0

    def observe(self, state: Any, action: Any, success: bool = False) -> bool:
        if success:
            self.reset()
            return False
        self._states.append(np.asarray(state, dtype=np.float32).copy())
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        if len(self._states) < int(self.window):
            return False
        self._states = self._states[-int(self.window) :]
        self._actions = self._actions[-int(self.window) :]
        states = np.asarray(self._states, dtype=np.float32)
        actions = np.asarray(self._actions, dtype=np.float32)
        state_span = float(np.max(np.linalg.norm(states - states[0], axis=1)))
        action_span = float(np.max(np.linalg.norm(actions - actions[0], axis=1)))
        if state_span <= float(self.state_epsilon) and action_span <= float(self.action_epsilon):
            self._stalled_windows += 1
        else:
            self._stalled_windows = 0
        return self._stalled_windows >= int(self.patience)


def _env_fps(env: Any) -> float:
    metadata = getattr(env, "metadata", {}) or {}
    fps = float(metadata.get("render_fps", 0.0) or 0.0)
    if fps > 0:
        return fps
    spec = getattr(env, "spec", None)
    if spec is not None:
        fps = float(getattr(spec, "effective_fps", 0.0) or 0.0)
    return fps if fps > 0 else 1.0


def _safe_state_digest(env: Any) -> Optional[Mapping[str, Any]]:
    client = getattr(env, "client", None)
    if client is None or not hasattr(client, "get_state_digest"):
        return None
    try:
        return client.get_state_digest()
    except Exception:
        return None


def _monitor_reason(info: Mapping[str, Any], collision_count: int, collision_limit: Optional[int]) -> Optional[str]:
    if bool(info.get("unsafe_state", False)):
        return UNSAFE_STATE
    if bool(info.get("object_lost", False)):
        return OBJECT_LOST
    if collision_limit is not None:
        current = collision_count + int(bool(info.get("collision", False) or info.get("contact_violation", False)))
        if current >= int(collision_limit):
            return COLLISION_LIMIT
    return None


def run_policy_rollout(
    env: Any,
    policy: Any,
    config: Any,
    artifact_writer: EpisodeArtifactWriter,
    seed: int,
    episode_index: int = 0,
) -> PolicyRolloutResult:
    """Run one direct ACT episode without planner or fallback."""

    policy.reset()
    queue = ACTActionQueue(
        chunk_size=int(policy.chunk_size),
        execution_horizon=int(config.execution_horizon) if config.execution_horizon is not None else int(policy.chunk_size),
    )
    queue.reset()
    adapter = RoCoPolicyObservationAdapter(policy.input_features, task_instruction=config.task_instruction)
    monitor = NoProgressMonitor(
        window=int(config.no_progress_window),
        patience=int(config.no_progress_patience),
        state_epsilon=float(config.no_progress_state_epsilon),
        action_epsilon=float(config.no_progress_action_epsilon),
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_dim = int(env.action_space.shape[0])
    fps = _env_fps(env)

    artifact_writer.write_episode_config(
        {
            "seed": int(seed),
            "episode_index": int(episode_index),
            "config": config.to_dict() if hasattr(config, "to_dict") else {},
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
            "execution_horizon": queue.execution_horizon,
            "chunk_size": queue.chunk_size,
        }
    )
    artifact_writer.write_manifest(
        {
            "phase": 4,
            "mode": "direct_policy_rollout",
            "planner_connected": False,
            "fallback_connected": False,
            "success_source": "info['is_success'] task predicate",
            "policy_metadata": dict(getattr(policy, "metadata", {})),
        }
    )

    inference_latencies: List[float] = []
    action_bound_violations = 0
    no_progress_events = 0
    collision_count = 0
    final_info: Dict[str, Any] = {"is_success": False}
    termination_reason = MAX_STEPS
    terminated = False
    truncated = False
    success = False
    num_steps = 0
    obs: Mapping[str, Any] = {}

    try:
        obs, reset_info = env.reset(seed=int(seed))
        final_info = dict(reset_info or {})
        final_info["is_success"] = bool(final_info.get("is_success", False))
        artifact_writer.append_event({"event": "RESET", "seed": int(seed), "is_success": final_info["is_success"]})
        if final_info["is_success"]:
            success = True
            terminated = True
            termination_reason = SUCCESS

        while not terminated and not truncated and num_steps < int(config.max_steps):
            if len(queue) == 0:
                try:
                    policy_inputs = adapter.to_policy_inputs(obs)
                    native_chunk, trace = policy.predict_native_chunk(
                        policy_inputs,
                        action_low=action_low,
                        action_high=action_high,
                        tolerance=float(config.action_bound_tolerance),
                    )
                except (ProcessorContractError, PolicyInferenceError, ActionQueueError) as exc:
                    termination_reason = POLICY_ERROR
                    final_info["error"] = str(exc)
                    terminated = True
                    break
                inference_latencies.append(float(trace.latency_ms))
                validation = validate_native_action_chunk(
                    native_chunk,
                    action_low=action_low,
                    action_high=action_high,
                    expected_action_dim=action_dim,
                    tolerance=float(config.action_bound_tolerance),
                )
                if not validation.finite:
                    termination_reason = NONFINITE_ACTION
                    final_info["action_validation"] = validation.to_dict()
                    terminated = True
                    break
                if not validation.shape_ok:
                    termination_reason = POLICY_ERROR
                    final_info["action_validation"] = validation.to_dict()
                    terminated = True
                    break
                action_bound_violations += validation.violation_count
                if validation.violation_count:
                    termination_reason = ACTION_OUT_OF_BOUNDS
                    final_info["action_validation"] = validation.to_dict()
                    terminated = True
                    break
                chunk_id = queue.load_chunk(native_chunk)
                artifact_writer.append_event(
                    {
                        "event": "POLICY_CHUNK",
                        "chunk_id": int(chunk_id),
                        "latency_ms": float(trace.latency_ms),
                        "processor_trace": adapter.last_trace.to_dict() if adapter.last_trace else {},
                        "inference_trace": trace.to_dict(),
                    }
                )

            queued = queue.pop()
            validation = validate_native_action_chunk(
                queued.action.reshape((1, -1)),
                action_low=action_low,
                action_high=action_high,
                expected_action_dim=action_dim,
                tolerance=float(config.action_bound_tolerance),
            )
            if not validation.finite:
                termination_reason = NONFINITE_ACTION
                final_info["action_validation"] = validation.to_dict()
                terminated = True
                break
            if validation.violation_count:
                action_bound_violations += validation.violation_count
                termination_reason = ACTION_OUT_OF_BOUNDS
                final_info["action_validation"] = validation.to_dict()
                terminated = True
                break

            try:
                next_obs, reward, step_terminated, step_truncated, info = env.step(queued.action)
            except Exception as exc:
                termination_reason = BRIDGE_ERROR
                final_info["error"] = str(exc)
                terminated = True
                break

            info = dict(info or {})
            info["is_success"] = bool(info.get("is_success", False))
            final_info = info
            num_steps += 1
            rendered = None
            if bool(config.record_video) and (num_steps % int(config.render_every_steps) == 0):
                try:
                    rendered = env.render()
                except Exception as exc:
                    artifact_writer.append_event({"event": "RENDER_FAILED", "env_step": num_steps, "error": str(exc)})
            artifact_writer.record_transition(
                env_step=num_steps,
                observation=next_obs,
                action=queued.action,
                queued_action=queued,
                reward=float(reward),
                info=info,
                rendered_frame=rendered,
            )
            trace = queued.trace_dict(env_step=num_steps)
            trace["is_success"] = info["is_success"]
            artifact_writer.append_chunk_trace(trace)

            monitor_reason = _monitor_reason(info, collision_count, config.collision_limit)
            collision_count += int(bool(info.get("collision", False) or info.get("contact_violation", False)))
            if monitor_reason is not None:
                termination_reason = monitor_reason
                terminated = True
            elif info["is_success"]:
                success = True
                termination_reason = SUCCESS
                terminated = True
            elif bool(step_terminated):
                termination_reason = SUCCESS if info["is_success"] else UNSAFE_STATE
                success = bool(info["is_success"])
                terminated = True
            elif bool(step_truncated):
                termination_reason = MAX_STEPS
                truncated = True
            else:
                stalled = monitor.observe(next_obs.get("agent_pos"), queued.action, success=False)
                if stalled:
                    no_progress_events += 1
                    termination_reason = NO_PROGRESS
                    terminated = True
            obs = next_obs

        if not terminated and not truncated and num_steps >= int(config.max_steps):
            termination_reason = MAX_STEPS
            truncated = True
    except KeyboardInterrupt:
        termination_reason = MANUAL_INTERRUPT
        terminated = True
    finally:
        queue.clear()

    final_digest = _safe_state_digest(env)
    result = PolicyRolloutResult(
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=termination_reason,
        num_env_steps=int(num_steps),
        sim_time=float(num_steps) / fps,
        inference_latency_ms=inference_latencies,
        action_bound_violations=int(action_bound_violations),
        no_progress_events=int(no_progress_events),
        final_info=final_info,
        artifact_dir=artifact_writer.artifact_dir,
    )
    artifact_writer.finalize(result.to_dict(), final_state_digest=final_digest, fps=fps)
    return result

