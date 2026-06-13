"""Production learned skill executor skeleton for Phase 5."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from integrations.lerobot_roco.evaluation.action_queue import ACTActionQueue

from rocobench.skills.executor import SkillExecutor
from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus, SkillPlan
from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT

from .artifacts import LearnedSkillArtifactWriter
from .config import LearnedExecutorConfig
from .errors import LearnedExecutionError, PolicyHandleError, PolicyRegistryError
from .fallback import FallbackController
from .models import (
    LearnedSkillExecutionStatus,
    LearnedSkillResultMetadata,
    MonitorEvent,
    MutableCancellationToken,
    ProgressStage,
    StateTransitionLog,
)
from .monitors import (
    EVENT_ACTION_OUT_OF_BOUNDS,
    EVENT_BRIDGE_FAILURE,
    EVENT_MANUAL_INTERRUPT,
    EVENT_NO_PROGRESS,
    EVENT_NONFINITE_ACTION,
    EVENT_POLICY_INFERENCE_FAILURE,
    EVENT_TIMEOUT,
    NoProgressMonitor,
    infer_progress_stage,
    monitor_info_events,
    validate_action,
)
from .policy_handle import BoundedPolicyHandleCache, NativeActionChunk
from .success import StableSkillSuccessChecker


LOGGER = logging.getLogger(__name__)


class DefaultLearnedEnvAdapter(object):
    """Small adapter for tests and typed client objects.

    A real runtime-separated deployment should provide an adapter whose
    ``step`` method talks to the Phase 0 bridge/client using native actions.
    """

    def action_bounds(self, env):
        if hasattr(env, "action_space"):
            return np.asarray(env.action_space.low, dtype=np.float32), np.asarray(env.action_space.high, dtype=np.float32)
        if hasattr(env, "action_low") and hasattr(env, "action_high"):
            return np.asarray(env.action_low, dtype=np.float32), np.asarray(env.action_high, dtype=np.float32)
        raise LearnedExecutionError("ACTION_BOUNDS_MISSING", "Learned env adapter cannot determine action bounds.")

    def state_vector(self, obs):
        if isinstance(obs, dict) and "agent_pos" in obs:
            return np.asarray(obs["agent_pos"], dtype=np.float32)
        if hasattr(obs, "agent_pos"):
            return np.asarray(obs.agent_pos, dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)

    def policy_observation(self, obs):
        return obs

    def step(self, env, action):
        return env.step(action)

    def render(self, env):
        if hasattr(env, "render"):
            return env.render()
        return None


def render_instruction(call):
    obj = call.arguments.get("object", "")
    container = call.arguments.get("container", "")
    canonical = "{}(object={}, container={})".format(call.skill_name, obj, container)
    instruction = "Put the {} into the {} bin slot.".format(obj, container.replace("bin_", "").replace("_", "-"))
    return {
        "canonical": canonical,
        "instruction": instruction,
        "template_version": "phase5.pack_put.v1",
        "language_conditioning_claimed": False,
    }


class LearnedSkillExecutor(SkillExecutor):
    def __init__(
        self,
        env,
        policy_registry,
        policy_cache=None,
        policy_loader=None,
        config=None,
        env_adapter=None,
        fallback_controller=None,
        cancellation_token=None,
    ):
        self.env = env
        self.policy_registry = policy_registry
        self.config = config or LearnedExecutorConfig()
        if policy_cache is None:
            if policy_loader is None:
                raise ValueError("policy_loader or policy_cache is required")
            policy_cache = BoundedPolicyHandleCache(policy_loader, max_size=self.config.policy_cache_size)
        self.policy_cache = policy_cache
        self.env_adapter = env_adapter or DefaultLearnedEnvAdapter()
        self.fallback_controller = fallback_controller or FallbackController(self.config.fallback)
        self.cancellation_token = cancellation_token or MutableCancellationToken()
        self._active = False

    def _transition(self, log, writer, state, evidence=None):
        item = log.append(state, evidence=evidence)
        writer.append_jsonl("state_machine.jsonl", item)
        return item

    def _select_active_call(self, plan):
        active = [call for call in plan.calls if call.skill_name != WAIT]
        if len(active) != 1:
            raise LearnedExecutionError(
                "INVALID_ACTIVE_CALL_COUNT",
                "Exactly one non-WAIT learned skill call is supported in Phase 5.",
                evidence={"active_call_count": len(active)},
            )
        for call in plan.calls:
            if call is not active[0] and call.skill_name != WAIT:
                raise LearnedExecutionError("PASSIVE_AGENT_NOT_WAIT", "Passive agents must WAIT.")
        if active[0].skill_name != PUT_OBJECT_IN_CONTAINER:
            raise LearnedExecutionError("UNSUPPORTED_SKILL", "Phase 5 supports PUT_OBJECT_IN_CONTAINER only.")
        return active[0]

    def _embodiment_id(self, call):
        mapping = getattr(self.env, "robot_name_map_inv", {})
        return str(mapping.get(call.agent_name, self.config.embodiment_id))

    def _make_result(
        self,
        success,
        status,
        reason,
        num_steps,
        reward,
        done,
        info,
        metadata,
    ):
        md = metadata.to_dict() if hasattr(metadata, "to_dict") else dict(metadata or {})
        return SkillExecutionResult(
            success=bool(success),
            status=status,
            reason=reason,
            num_sim_steps=int(num_steps),
            reward=float(reward),
            done=bool(done),
            info=dict(info or {}),
            metadata=md,
        )

    def execute(self, plan: SkillPlan, obs, artifact_dir: Optional[str] = None) -> SkillExecutionResult:
        if self._active:
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.EXECUTION_FAILED,
                reason="Another learned skill call is already active.",
                num_sim_steps=0,
                reward=0.0,
                done=False,
                info={},
                metadata={"failure_code": "ACTIVE_CALL_IN_PROGRESS"},
            )
        self._active = True
        writer = LearnedSkillArtifactWriter(artifact_dir)
        state_log = StateTransitionLog()
        monitor_events = []
        fallback_decision = None
        active_call = None
        spec = None
        progress_stage = ProgressStage.NOT_STARTED.value
        num_steps = 0
        reward = 0.0
        done = False
        info = {}
        failure_code = None
        learned_success = False

        try:
            self._transition(state_log, writer, LearnedSkillExecutionStatus.CREATED, {"plan_id": plan.plan_id})
            active_call = self._select_active_call(plan)
            instruction = render_instruction(active_call)
            writer.write_json("skill_call.json", active_call.to_dict())
            writer.write_json("instruction.json", instruction)
            writer.write_json("executor_config.json", self.config.to_dict())

            self._transition(state_log, writer, LearnedSkillExecutionStatus.VALIDATING)
            embodiment = self._embodiment_id(active_call)
            spec = self.policy_registry.resolve(
                active_call.skill_name,
                active_call.agent_name,
                embodiment,
                self.config.task_id,
            )
            self.policy_registry.validate_static(spec)
            writer.write_json("policy_spec.json", spec.to_dict())

            self._transition(state_log, writer, LearnedSkillExecutionStatus.LOADING_POLICY, {"policy_id": spec.policy_id})
            handle = self.policy_cache.get(spec)
            checkpoint_metadata = handle.health_check(spec)
            writer.write_json("checkpoint_metadata.json", checkpoint_metadata)

            self._transition(state_log, writer, LearnedSkillExecutionStatus.RESETTING_POLICY)
            handle.reset()
            queue = None
            success_checker = StableSkillSuccessChecker(self.config.stable_success_checks)
            no_progress = NoProgressMonitor(
                self.config.no_progress_window,
                self.config.no_progress_patience,
                self.config.no_progress_state_epsilon,
                self.config.no_progress_action_epsilon,
            )
            action_low, action_high = self.env_adapter.action_bounds(self.env)

            self._transition(state_log, writer, LearnedSkillExecutionStatus.RUNNING)
            current_obs = obs
            while num_steps < int(spec.max_steps):
                self.cancellation_token.throw_if_cancelled()
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
                        event = MonitorEvent(failure_code, "error", str(exc), {}, "Request fallback or abort.")
                        monitor_events.append(event)
                        writer.append_jsonl("monitor_events.jsonl", event.to_dict())
                        break
                    if not isinstance(chunk, NativeActionChunk):
                        chunk = NativeActionChunk(np.asarray(chunk, dtype=np.float32), {})
                    writer.append_jsonl("policy_chunks.jsonl", chunk.to_dict())
                    if queue is None or queue.chunk_size != int(chunk.actions.shape[0]):
                        queue = ACTActionQueue(int(chunk.actions.shape[0]), spec.execution_horizon)
                    queue.load_chunk(chunk.actions)

                queued = queue.pop()
                event = validate_action(queued.action, action_low, action_high, self.config.action_bound_tolerance)
                if event is not None:
                    failure_code = event.code
                    monitor_events.append(event)
                    writer.append_jsonl("monitor_events.jsonl", event.to_dict())
                    break

                try:
                    next_obs, reward, done, info = self.env_adapter.step(self.env, queued.action)
                except Exception as exc:
                    failure_code = EVENT_BRIDGE_FAILURE
                    event = MonitorEvent(failure_code, "error", str(exc), {}, "Abort learned execution.")
                    monitor_events.append(event)
                    writer.append_jsonl("monitor_events.jsonl", event.to_dict())
                    break

                num_steps += 1
                state = self.env_adapter.state_vector(next_obs)
                writer.record_step(state, queued.action, queued.chunk_id, queued.chunk_offset)
                for event in monitor_info_events(info):
                    monitor_events.append(event)
                    writer.append_jsonl("monitor_events.jsonl", event.to_dict())
                    if event.severity == "error":
                        failure_code = event.code
                        break
                if failure_code:
                    current_obs = next_obs
                    break

                progress_stage = infer_progress_stage(
                    self.env,
                    next_obs,
                    active_call.agent_name,
                    active_call.arguments["object"],
                    active_call.arguments["container"],
                ).value
                if success_checker.check(
                    self.env,
                    next_obs,
                    active_call.agent_name,
                    active_call.arguments["object"],
                    active_call.arguments["container"],
                ):
                    self._transition(state_log, writer, LearnedSkillExecutionStatus.VERIFYING, success_checker.last_evidence)
                    learned_success = True
                    progress_stage = success_checker.progress_stage()
                    current_obs = next_obs
                    break

                event = no_progress.observe(state, queued.action)
                if event is not None and event.code in spec.failure_monitors:
                    failure_code = event.code
                    monitor_events.append(event)
                    writer.append_jsonl("monitor_events.jsonl", event.to_dict())
                    current_obs = next_obs
                    break
                current_obs = next_obs

            if learned_success:
                final_state = LearnedSkillExecutionStatus.SUCCEEDED
                status = SkillExecutionStatus.SUCCESS
                reason = ""
            elif failure_code is None and num_steps >= int(spec.max_steps):
                failure_code = EVENT_TIMEOUT
                event = MonitorEvent(EVENT_TIMEOUT, "error", "Learned skill exceeded max_steps.", {"max_steps": spec.max_steps}, "Request fallback or replan.")
                monitor_events.append(event)
                writer.append_jsonl("monitor_events.jsonl", event.to_dict())
                final_state = LearnedSkillExecutionStatus.TIMED_OUT
                status = SkillExecutionStatus.TIMEOUT
                reason = event.message
            else:
                final_state = LearnedSkillExecutionStatus.FAILED
                status = SkillExecutionStatus.EXECUTION_FAILED
                reason = monitor_events[-1].message if monitor_events else "Learned skill failed."

            fallback_decision = self.fallback_controller.decide(
                failure_code,
                plan=plan,
                obs=current_obs,
                artifact_dir=artifact_dir,
            ) if not learned_success else None
            if fallback_decision is not None and fallback_decision.recommended:
                self._transition(state_log, writer, LearnedSkillExecutionStatus.FALLBACK_REQUESTED, fallback_decision.to_dict())
            self._transition(state_log, writer, final_state, {"failure_code": failure_code})
            self._transition(state_log, writer, LearnedSkillExecutionStatus.CLOSED)

            md = LearnedSkillResultMetadata(
                policy_id=spec.policy_id if spec else "",
                policy_revision=spec.checkpoint_revision if spec else "",
                learned_backend=spec.policy_type if spec else "unknown",
                failure_code=failure_code,
                progress_stage=progress_stage,
                monitor_events=tuple(monitor_events),
                fallback_recommended=bool(fallback_decision and fallback_decision.recommended),
                fallback_used=bool(fallback_decision and fallback_decision.used),
                fallback_success=bool(fallback_decision and fallback_decision.success),
                learned_success=bool(learned_success),
                artifact_path=artifact_dir,
                state_transitions=tuple(state_log.transitions),
                evidence={"success_evidence": success_checker.last_evidence},
            )
            writer.write_json("fallback.json", fallback_decision.to_dict() if fallback_decision else {"recommended": False, "used": False, "success": False})
            writer.finalize_arrays()
            result = self._make_result(
                learned_success or bool(fallback_decision and fallback_decision.success),
                status if not (fallback_decision and fallback_decision.success) else SkillExecutionStatus.SUCCESS,
                reason,
                num_steps,
                reward,
                done,
                info,
                md,
            )
            writer.write_json("result.json", result.to_dict())
            return result
        except PolicyRegistryError as exc:
            failure_code = exc.code
            event = MonitorEvent(exc.code, "error", exc.message, exc.evidence, "Fix learned policy registry.")
            monitor_events.append(event)
            writer.append_jsonl("monitor_events.jsonl", event.to_dict())
            self._transition(state_log, writer, LearnedSkillExecutionStatus.FAILED, {"failure_code": failure_code})
            self._transition(state_log, writer, LearnedSkillExecutionStatus.CLOSED)
            md = {
                "failure_code": failure_code,
                "monitor_events": [event.to_dict() for event in monitor_events],
                "state_transitions": list(state_log.transitions),
            }
            result = SkillExecutionResult(False, SkillExecutionStatus.INVALID_PLAN, exc.message, num_steps, reward, done, info, md)
            writer.write_json("result.json", result.to_dict())
            return result
        except LearnedExecutionError as exc:
            failure_code = exc.code
            status = SkillExecutionStatus.INTERRUPTED if exc.code == EVENT_MANUAL_INTERRUPT else SkillExecutionStatus.EXECUTION_FAILED
            event = MonitorEvent(exc.code, "error", exc.message, exc.evidence, "Abort learned execution.")
            monitor_events.append(event)
            writer.append_jsonl("monitor_events.jsonl", event.to_dict())
            self._transition(state_log, writer, LearnedSkillExecutionStatus.INTERRUPTED if status == SkillExecutionStatus.INTERRUPTED else LearnedSkillExecutionStatus.FAILED, {"failure_code": failure_code})
            self._transition(state_log, writer, LearnedSkillExecutionStatus.CLOSED)
            md = {
                "failure_code": failure_code,
                "monitor_events": [event.to_dict() for event in monitor_events],
                "state_transitions": list(state_log.transitions),
            }
            result = SkillExecutionResult(False, status, exc.message, num_steps, reward, done, info, md)
            writer.write_json("result.json", result.to_dict())
            return result
        finally:
            self._active = False
