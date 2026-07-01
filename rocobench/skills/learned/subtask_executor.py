"""SubtaskLearnedExecutor — generalised learned-policy executor for all subtasks.

Extends the existing LearnedSkillExecutor machinery beyond PUT_OBJECT_IN_CONTAINER
to support PICK, PLACE, PICK_AND_PLACE, OPEN_CABINET, STACK_ON, SWEEP.

Key differences from LearnedSkillExecutor:
  - Skill name is not hardcoded — any registered SUBTASK_SKILL is accepted.
  - Success checking is delegated to SubtaskSuccessChecker, one per skill.
  - env_adapter step converts flat ctrl arrays → SimAction (no LeRobot gym wrapper needed).
  - Uncertainty estimation is pluggable at construction time.
  - All artefacts (chunks, events, uncertainty trace) are written to artifact_dir.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from rocobench.crie_bt.uncertainty import UncertaintyEstimator
from rocobench.crie_bt.types import ExecutionFeedback, SkillCall as BTSkillCall
from rocobench.envs.base_env import SimAction
from rocobench.skills.executor import SkillExecutor
from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus, SkillPlan

from .artifacts import LearnedSkillArtifactWriter
from .errors import LearnedExecutionError, PolicyHandleError, PolicyRegistryError
from .fallback import FallbackController
from .models import (
    LearnedSkillExecutionStatus,
    LearnedSkillResultMetadata,
    MonitorEvent,
    MutableCancellationToken,
    StateTransitionLog,
)
from .monitors import (
    EVENT_BRIDGE_FAILURE,
    EVENT_MANUAL_INTERRUPT,
    EVENT_NO_PROGRESS,
    EVENT_NONFINITE_ACTION,
    EVENT_POLICY_INFERENCE_FAILURE,
    EVENT_TIMEOUT,
    NoProgressMonitor,
    validate_action,
)
from .policy_handle import BoundedPolicyHandleCache, NativeActionChunk
from .subtask_skills import SKILL_WAIT, ALL_SUBTASK_SKILLS
from .subtask_success import SubtaskSuccessChecker, build_success_checker

try:
    from integrations.lerobot_roco.evaluation.action_queue import ACTActionQueue
except ImportError:  # pragma: no cover — optional dep
    ACTActionQueue = None  # type: ignore

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimAction env adapter — no Gymnasium needed
# ---------------------------------------------------------------------------

class SubtaskEnvAdapter:
    """
    Bridges the flat numpy ctrl array produced by the policy to the SimAction
    expected by MujocoSimEnv.step().

    The action encoding is:  action[i] = ctrl value for ctrl_idx[i]
    where ctrl_idxs are pre-collected from all robots in order.
    """

    def __init__(self, env, robots: Dict[str, Any]):
        self.env = env
        self.robots = robots
        self._ctrl_idxs: List[int] = []
        self._qpos_idxs: List[int] = []
        for robot in robots.values():
            self._ctrl_idxs.extend(robot.joint_idxs_in_ctrl)
            self._qpos_idxs.extend(robot.joint_idxs_in_qpos)
        # obs attributes use hardware body names, not agent display names
        name_map_inv = getattr(env, "robot_name_map_inv", {})
        self._obs_robot_names: List[str] = [
            name_map_inv.get(n, n) for n in robots
        ]

    def action_bounds(self, env) -> Tuple[np.ndarray, np.ndarray]:
        """Joint ctrl limits from the MuJoCo model."""
        n = len(self._ctrl_idxs)
        low = np.full(n, -np.pi, dtype=np.float32)
        high = np.full(n, np.pi, dtype=np.float32)
        try:
            for i, ctrl_idx in enumerate(self._ctrl_idxs):
                lo, hi = env.physics.model.actuator_ctrlrange[ctrl_idx]
                if lo < hi:
                    low[i] = float(lo)
                    high[i] = float(hi)
        except Exception:
            pass
        return low, high

    def policy_observation(self, obs):
        """Pass through — mock handle reads obs directly; real ACT would render images."""
        return obs

    def state_vector(self, obs) -> np.ndarray:
        """Flat EE positions for no-progress monitoring."""
        pos = []
        for robot_name in self._obs_robot_names:
            agent = getattr(obs, robot_name, None)
            if agent is not None:
                pos.extend(agent.ee_xpos.tolist())
        return np.array(pos, dtype=np.float32) if pos else np.zeros(3, dtype=np.float32)

    def step(self, env, action: np.ndarray,
             eq_active_idxs=None, eq_active_vals=None):
        """Convert flat ctrl array → SimAction → env.step().

        action[i] is the ctrl value for self._ctrl_idxs[i].
        eq_active_idxs/vals carry weld/gripper equality-constraint commands
        (used by mock handle to fire the weld on the grasp step).
        """
        action = np.asarray(action, dtype=np.float32)
        n_qpos = len(self._qpos_idxs)
        qpos_target = action[:n_qpos] if len(action) >= n_qpos else np.zeros(n_qpos, dtype=np.float32)
        sim_action = SimAction(
            ctrl_vals=action,
            ctrl_idxs=np.array(self._ctrl_idxs, dtype=np.int32),
            qpos_idxs=np.array(self._qpos_idxs, dtype=np.int32),
            qpos_target=qpos_target,
            eq_active_idxs=eq_active_idxs,
            eq_active_vals=eq_active_vals,
        )
        return env.step(sim_action, verbose=False)

    def render(self, env):
        return None


# ---------------------------------------------------------------------------
# Instruction rendering
# ---------------------------------------------------------------------------

def render_subtask_instruction(call) -> Dict[str, Any]:
    """Build instruction dict from a SkillPlan call (works with both SkillCall types)."""
    skill_name = getattr(call, "skill_name", "") or getattr(call, "skill_name", "")
    args = dict(getattr(call, "arguments", {}))
    canonical_parts = [f"{k}={v}" for k, v in sorted(args.items())]
    canonical = f"{skill_name}({', '.join(canonical_parts)})"
    # Human-readable form
    if skill_name == "PICK":
        text = f"Pick up the {args.get('object', '?')}."
    elif skill_name == "PLACE":
        text = f"Place it at {args.get('target', '?')}."
    elif skill_name == "PICK_AND_PLACE":
        text = f"Pick the {args.get('object', '?')} and place it at {args.get('target', '?')}."
    elif skill_name == "STACK_ON":
        text = f"Stack it on top of the {args.get('target', '?')}."
    elif skill_name == "OPEN_CABINET":
        text = f"Open the {args.get('door', '?')}."
    else:
        text = canonical
    return {
        "canonical": canonical,
        "instruction": text,
        "skill_name": skill_name,
        **args,
    }


# ---------------------------------------------------------------------------
# SubtaskLearnedExecutor
# ---------------------------------------------------------------------------

class SubtaskLearnedExecutor(SkillExecutor):
    """
    Learned-policy executor supporting all subtask skill types.

    Parameters
    ----------
    env : MujocoSimEnv
    robots : dict[str, SimRobot]
    policy_registry : LearnedPolicyRegistry
    policy_cache : BoundedPolicyHandleCache | None
    policy_loader : callable | None  — used if policy_cache is None
    config : LearnedExecutorConfig
    uncertainty_mode : str  — "none" | "heuristic" | "policy_metadata" | "ensemble_variance"
    stable_success_checks : int  — consecutive checks before declaring success
    max_steps : int  — per-skill step budget
    """

    def __init__(
        self,
        env,
        robots: Dict[str, Any],
        policy_registry,
        policy_cache=None,
        policy_loader=None,
        config=None,
        uncertainty_mode: str = "policy_metadata",
        stable_success_checks: int = 2,
        max_steps: int = 300,
        cancellation_token=None,
        fallback_controller=None,
    ):
        self.env = env
        self.robots = robots
        self.policy_registry = policy_registry
        self.uncertainty_mode = uncertainty_mode
        self.stable_success_checks = int(stable_success_checks)
        self.max_steps = int(max_steps)
        self.cancellation_token = cancellation_token or MutableCancellationToken()
        self.fallback_controller = fallback_controller or FallbackController()

        from .config import LearnedExecutorConfig
        self.config = config or LearnedExecutorConfig()

        if policy_cache is None:
            if policy_loader is None:
                raise ValueError("Provide policy_cache or policy_loader.")
            policy_cache = BoundedPolicyHandleCache(policy_loader, max_size=self.config.policy_cache_size)
        self.policy_cache = policy_cache

        self.env_adapter = SubtaskEnvAdapter(env, robots)
        self._active = False

    # ------------------------------------------------------------------
    # SkillExecutor interface
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: SkillPlan,
        obs,
        artifact_dir: Optional[str] = None,
    ) -> SkillExecutionResult:
        if self._active:
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.EXECUTION_FAILED,
                reason="Another skill is already active.",
                num_sim_steps=0, reward=0.0, done=False, info={},
                metadata={"failure_code": "ACTIVE_CALL_IN_PROGRESS"},
            )
        self._active = True
        writer = LearnedSkillArtifactWriter(artifact_dir)
        state_log = StateTransitionLog()

        try:
            return self._run(plan, obs, writer, state_log, artifact_dir)
        finally:
            self._active = False

    # ------------------------------------------------------------------
    # Internal execution loop
    # ------------------------------------------------------------------

    def _run(self, plan, obs, writer, state_log, artifact_dir):
        monitor_events: List[MonitorEvent] = []
        failure_code: Optional[str] = None
        num_steps = 0
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        learned_success = False
        spec = None
        fallback_decision = None

        try:
            self._t(state_log, writer, LearnedSkillExecutionStatus.CREATED, {"plan_id": plan.plan_id})

            # Select the active (non-WAIT) skill call
            active_call = self._select_active_call(plan)
            skill_name = active_call.skill_name
            call_args = dict(getattr(active_call, "arguments", {}))
            instruction = render_subtask_instruction(active_call)

            writer.write_json("skill_call.json", active_call.to_dict())
            writer.write_json("instruction.json", instruction)

            self._t(state_log, writer, LearnedSkillExecutionStatus.VALIDATING)

            # Resolve policy spec
            embodiment = self._embodiment_id(active_call)
            spec = self.policy_registry.resolve(
                skill_name, active_call.agent_name, embodiment, self.config.task_id,
            )
            self.policy_registry.validate_static(spec)
            writer.write_json("policy_spec.json", spec.to_dict())

            # Load policy
            self._t(state_log, writer, LearnedSkillExecutionStatus.LOADING_POLICY,
                    {"policy_id": spec.policy_id})
            handle = self.policy_cache.get(spec)
            checkpoint_meta = handle.health_check(spec)
            writer.write_json("checkpoint_metadata.json", checkpoint_meta)

            # Reset policy and monitors
            self._t(state_log, writer, LearnedSkillExecutionStatus.RESETTING_POLICY)
            handle.reset()

            success_checker = build_success_checker(skill_name, self.stable_success_checks)
            uncertainty_estimator = UncertaintyEstimator(self.uncertainty_mode)
            no_progress = NoProgressMonitor(
                self.config.no_progress_window,
                self.config.no_progress_patience,
                self.config.no_progress_state_epsilon,
                self.config.no_progress_action_epsilon,
            )

            action_low, action_high = self.env_adapter.action_bounds(self.env)
            queue = None
            _eq_active_queue: list = []  # parallel queue for weld commands

            self._t(state_log, writer, LearnedSkillExecutionStatus.RUNNING)
            current_obs = obs
            uncertainty_trace: List[Dict] = []

            # ---- main loop ----
            while num_steps < self.max_steps:
                self.cancellation_token.throw_if_cancelled()

                # Predict new chunk when queue is empty
                if queue is None or len(queue) == 0:
                    try:
                        chunk = handle.predict_native_chunk(
                            self.env_adapter.policy_observation(current_obs),
                            instruction,
                            action_low,
                            action_high,
                        )
                    except Exception as exc:
                        failure_code = EVENT_POLICY_INFERENCE_FAILURE
                        monitor_events.append(MonitorEvent(
                            failure_code, "error", str(exc), {}, "Request fallback."))
                        break

                    if not isinstance(chunk, NativeActionChunk):
                        chunk = NativeActionChunk(np.asarray(chunk, dtype=np.float32), {})
                    writer.append_jsonl("policy_chunks.jsonl", chunk.to_dict())

                    # Estimate uncertainty from this chunk
                    bt_call = BTSkillCall(
                        agent=active_call.agent_name,
                        skill_name=skill_name,
                        arguments=call_args,
                    )
                    unc_state = uncertainty_estimator.estimate(
                        bt_call, current_obs, action_chunk=chunk.actions
                    )
                    uncertainty_trace.append({
                        "step": num_steps,
                        "confidence": unc_state.confidence,
                        "uncertainty": unc_state.uncertainty,
                        "risk_level": unc_state.risk_level,
                        "source": unc_state.source,
                    })
                    writer.append_jsonl("uncertainty_trace.jsonl", uncertainty_trace[-1])

                    # Abort if risk is critically high after first few steps
                    if num_steps > 10 and unc_state.risk_level == "high":
                        LOGGER.warning(
                            "[SubtaskExecutor] High uncertainty at step %d "
                            "(confidence=%.2f) for %s — continuing but flagging.",
                            num_steps, unc_state.confidence, skill_name,
                        )

                    chunk_size = int(chunk.actions.shape[0])
                    if ACTActionQueue is not None:
                        if queue is None or queue.chunk_size != chunk_size:
                            queue = ACTActionQueue(chunk_size, spec.execution_horizon)
                        queue.load_chunk(chunk.actions)
                    else:
                        # Fallback: use a simple list queue
                        queue = _SimpleQueue(chunk.actions, spec.execution_horizon)

                    # Collect per-step eq_active (weld/gripper) commands from chunk metadata
                    eq_per_step = chunk.metadata.get("eq_active_per_step") if chunk.metadata else None
                    if eq_per_step:
                        _eq_active_queue.extend(eq_per_step)

                queued = queue.pop()
                action = np.asarray(queued.action if hasattr(queued, "action") else queued,
                                    dtype=np.float32)

                # Pop eq_active for this step (if any)
                eq_entry = _eq_active_queue.pop(0) if _eq_active_queue else None
                eq_idxs = np.array(eq_entry[0], dtype=np.int32) if eq_entry else None
                eq_vals = np.array(eq_entry[1], dtype=np.int32) if eq_entry else None

                # Validate
                event = validate_action(action, action_low, action_high,
                                        self.config.action_bound_tolerance)
                if event is not None:
                    failure_code = event.code
                    monitor_events.append(event)
                    break

                # Step env
                try:
                    next_obs, reward, done, info = self.env_adapter.step(
                        self.env, action, eq_active_idxs=eq_idxs, eq_active_vals=eq_vals,
                    )
                except Exception as exc:
                    failure_code = EVENT_BRIDGE_FAILURE
                    monitor_events.append(MonitorEvent(
                        failure_code, "error", str(exc), {}, "Abort."))
                    break

                num_steps += 1
                state = self.env_adapter.state_vector(next_obs)

                # Check success
                if success_checker.check(self.env, next_obs, call_args):
                    self._t(state_log, writer, LearnedSkillExecutionStatus.VERIFYING,
                            success_checker.last_evidence)
                    learned_success = True
                    current_obs = next_obs
                    break

                # No-progress monitor
                event = no_progress.observe(state, action)
                if event is not None and event.code in spec.failure_monitors:
                    failure_code = event.code
                    monitor_events.append(event)
                    current_obs = next_obs
                    break

                current_obs = next_obs
                if done:
                    break

            # ---- resolve final state ----
            if learned_success:
                final_state = LearnedSkillExecutionStatus.SUCCEEDED
                status = SkillExecutionStatus.SUCCESS
                reason = ""
            elif failure_code is None and num_steps >= self.max_steps:
                failure_code = EVENT_TIMEOUT
                final_state = LearnedSkillExecutionStatus.TIMED_OUT
                status = SkillExecutionStatus.TIMEOUT
                reason = f"Exceeded {self.max_steps} steps."
            else:
                final_state = LearnedSkillExecutionStatus.FAILED
                status = SkillExecutionStatus.EXECUTION_FAILED
                reason = monitor_events[-1].message if monitor_events else "Skill failed."

            # Fallback decision
            fallback_decision = (
                self.fallback_controller.decide(failure_code, plan=plan, obs=current_obs,
                                                artifact_dir=artifact_dir)
                if not learned_success else None
            )
            self._t(state_log, writer, final_state, {"failure_code": failure_code})
            self._t(state_log, writer, LearnedSkillExecutionStatus.CLOSED)

            writer.write_json("uncertainty_summary.json",
                             _summarise_uncertainty(uncertainty_trace))
            writer.write_json("fallback.json",
                              fallback_decision.to_dict() if fallback_decision else {})
            writer.finalize_arrays()

            md = LearnedSkillResultMetadata(
                policy_id=spec.policy_id if spec else "",
                policy_revision=spec.checkpoint_revision if spec else "",
                learned_backend=spec.policy_type if spec else "mock",
                failure_code=failure_code,
                progress_stage=success_checker.progress_stage(),
                monitor_events=tuple(monitor_events),
                fallback_recommended=bool(fallback_decision and fallback_decision.recommended),
                fallback_used=bool(fallback_decision and fallback_decision.used),
                fallback_success=bool(fallback_decision and fallback_decision.success),
                learned_success=learned_success,
                artifact_path=artifact_dir,
                state_transitions=tuple(state_log.transitions),
                evidence={"success_evidence": success_checker.last_evidence,
                          "uncertainty_trace_len": len(uncertainty_trace)},
            )
            success = learned_success or bool(fallback_decision and fallback_decision.success)
            if fallback_decision and fallback_decision.success:
                status = SkillExecutionStatus.SUCCESS
            result = SkillExecutionResult(
                success=success, status=status, reason=reason,
                num_sim_steps=num_steps, reward=float(reward), done=bool(done),
                info=dict(info), metadata=md.to_dict(),
            )
            writer.write_json("result.json", result.to_dict())
            return result

        except PolicyRegistryError as exc:
            return self._error_result(exc.code, exc.message, num_steps, reward, done, info,
                                      state_log, writer, monitor_events,
                                      SkillExecutionStatus.INVALID_PLAN)
        except LearnedExecutionError as exc:
            st = (SkillExecutionStatus.INTERRUPTED
                  if exc.code == EVENT_MANUAL_INTERRUPT
                  else SkillExecutionStatus.EXECUTION_FAILED)
            return self._error_result(exc.code, exc.message, num_steps, reward, done, info,
                                      state_log, writer, monitor_events, st)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _t(self, log, writer, state, evidence=None):
        item = log.append(state, evidence=evidence)
        writer.append_jsonl("state_machine.jsonl", item)

    def _select_active_call(self, plan: SkillPlan):
        active = [c for c in plan.calls if c.skill_name != SKILL_WAIT]
        if len(active) != 1:
            raise LearnedExecutionError(
                "INVALID_ACTIVE_CALL_COUNT",
                f"Expected exactly one non-WAIT call, got {len(active)}.",
                evidence={"active_count": len(active)},
            )
        return active[0]

    def _embodiment_id(self, call) -> str:
        mapping = getattr(self.env, "robot_name_map_inv", {})
        agent_name = getattr(call, "agent_name", "")
        return str(mapping.get(agent_name, self.config.embodiment_id))

    def _error_result(self, code, msg, num_steps, reward, done, info,
                      log, writer, events, status):
        event = MonitorEvent(code, "error", msg, {}, "")
        events.append(event)
        writer.append_jsonl("monitor_events.jsonl", event.to_dict())
        self._t(log, writer, LearnedSkillExecutionStatus.FAILED, {"failure_code": code})
        self._t(log, writer, LearnedSkillExecutionStatus.CLOSED)
        return SkillExecutionResult(
            success=False, status=status, reason=msg,
            num_sim_steps=num_steps, reward=float(reward),
            done=bool(done), info=dict(info),
            metadata={"failure_code": code,
                      "monitor_events": [e.to_dict() for e in events]},
        )


# ---------------------------------------------------------------------------
# Simple queue fallback when ACTActionQueue is unavailable
# ---------------------------------------------------------------------------

class _QueuedAction:
    def __init__(self, action, chunk_id=0, chunk_offset=0):
        self.action = action
        self.chunk_id = chunk_id
        self.chunk_offset = chunk_offset


class _SimpleQueue:
    """Minimal action queue when lerobot_roco is not installed."""

    def __init__(self, actions: np.ndarray, execution_horizon: int):
        self.chunk_size = int(actions.shape[0])
        self.execution_horizon = int(execution_horizon)
        self._actions = list(actions)
        self._idx = 0
        self._chunk_id = 0

    def __len__(self):
        return max(0, min(self.execution_horizon, len(self._actions) - self._idx))

    def load_chunk(self, actions: np.ndarray):
        self._actions = list(actions)
        self._idx = 0
        self._chunk_id += 1

    def pop(self) -> _QueuedAction:
        if self._idx >= len(self._actions):
            raise IndexError("Queue empty")
        action = self._actions[self._idx]
        item = _QueuedAction(action, self._chunk_id, self._idx)
        self._idx += 1
        return item


# ---------------------------------------------------------------------------
# Uncertainty summary
# ---------------------------------------------------------------------------

def _summarise_uncertainty(trace: List[Dict]) -> Dict[str, Any]:
    if not trace:
        return {"n_chunks": 0}
    confidences = [t["confidence"] for t in trace]
    risks = [t["risk_level"] for t in trace]
    return {
        "n_chunks": len(trace),
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "high_risk_chunks": int(sum(r == "high" for r in risks)),
        "medium_risk_chunks": int(sum(r == "medium" for r in risks)),
        "low_risk_chunks": int(sum(r == "low" for r in risks)),
    }
