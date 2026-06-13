import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from .models import SkillExecutionResult, SkillExecutionStatus, SkillPlan


LOGGER = logging.getLogger(__name__)


class SkillExecutor(ABC):
    """Backend-independent interface for executing prepared skill plans."""

    @abstractmethod
    def execute(self, plan: SkillPlan, obs, artifact_dir: Optional[str] = None) -> SkillExecutionResult:
        pass


def _default_policy_factory(**kwargs):
    from rocobench.policy import PlannedPathPolicy

    return PlannedPathPolicy(**kwargs)


class RRTSkillExecutor(SkillExecutor):
    """Execute prepared RRT skill plans with PlannedPathPolicy."""

    def __init__(
        self,
        env,
        robots: Dict[str, Any],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        policy_factory: Optional[Callable[..., Any]] = None,
        plan_splitted: bool = False,
        max_sim_steps: int = 5000,
        video_format: str = "mp4",
        transition_observer: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.robots = robots
        self.policy_kwargs = dict(policy_kwargs or {})
        self.policy_factory = policy_factory or _default_policy_factory
        self.plan_splitted = plan_splitted
        self.max_sim_steps = max_sim_steps
        self.video_format = video_format
        self.transition_observer = transition_observer

    def execute(self, plan: SkillPlan, obs, artifact_dir: Optional[str] = None) -> SkillExecutionResult:
        if plan.prepared_execution is None:
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.NOT_PREPARED,
                reason="Skill plan has no prepared execution.",
                num_sim_steps=0,
                reward=0,
                done=False,
                info={},
            )
        prepared = plan.prepared_execution
        if prepared.backend_name != "rrt":
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.INVALID_PLAN,
                reason="Prepared backend is {}, expected rrt.".format(prepared.backend_name),
                num_sim_steps=0,
                reward=0,
                done=False,
                info={},
            )

        if artifact_dir is not None:
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, "compiled_path_plans.pkl"), "wb") as f:
                pickle.dump(prepared.compiled_plans, f)

        num_sim_steps = 0
        reward = 0
        done = False
        info = {}
        plan_reasons = []

        try:
            for index, path_plan in enumerate(prepared.compiled_plans):
                policy = self.policy_factory(
                    physics=self.env.physics,
                    robots=self.robots,
                    path_plan=path_plan,
                    graspable_object_names=self.env.get_graspable_objects(),
                    allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                    plan_splitted=self.plan_splitted,
                    **self.policy_kwargs
                )
                plan_success, reason = policy.plan(self.env)
                plan_reasons.append(reason)
                if not plan_success:
                    return SkillExecutionResult(
                        success=False,
                        status=SkillExecutionStatus.MOTION_PLANNING_FAILED,
                        reason=reason,
                        num_sim_steps=num_sim_steps,
                        reward=reward,
                        done=done,
                        info=info,
                        metadata={"plan_index": index, "plan_reasons": plan_reasons},
                    )

                if artifact_dir is not None:
                    with open(os.path.join(artifact_dir, "rrt_plan_{}.pkl".format(index)), "wb") as f:
                        pickle.dump(policy.rrt_plan_results, f)
                    with open(os.path.join(artifact_dir, "actions_{}.pkl".format(index)), "wb") as f:
                        pickle.dump(policy.action_buffer, f)

                while not policy.plan_exhausted:
                    if num_sim_steps >= self.max_sim_steps:
                        return SkillExecutionResult(
                            success=False,
                            status=SkillExecutionStatus.TIMEOUT,
                            reason="Exceeded maximum skill execution steps: {}.".format(self.max_sim_steps),
                            num_sim_steps=num_sim_steps,
                            reward=reward,
                            done=done,
                            info=info,
                            metadata={"plan_index": index, "plan_reasons": plan_reasons},
                        )
                    sim_action = policy.act(obs, self.env.physics)
                    if self.transition_observer is not None:
                        self.transition_observer.before_step(
                            obs,
                            sim_action,
                            {
                                "plan_index": index,
                                "skill_step_index": num_sim_steps,
                                "plan_id": plan.plan_id,
                            },
                        )
                    obs, reward, done, info = self.env.step(sim_action, verbose=False)
                    num_sim_steps += 1
                    if self.transition_observer is not None:
                        self.transition_observer.after_step(
                            {
                                "observation": obs,
                                "reward": reward,
                                "done": done,
                                "info": info,
                                "skill_step_index": num_sim_steps,
                                "plan_id": plan.plan_id,
                            }
                        )
                    if done:
                        break
                if done:
                    break
        except KeyboardInterrupt:
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.INTERRUPTED,
                reason="Execution interrupted.",
                num_sim_steps=num_sim_steps,
                reward=reward,
                done=done,
                info=info,
                metadata={"plan_reasons": plan_reasons},
            )
        except Exception as exc:
            LOGGER.exception("Skill execution failed for plan %s", plan.plan_id)
            return SkillExecutionResult(
                success=False,
                status=SkillExecutionStatus.EXECUTION_FAILED,
                reason=str(exc),
                num_sim_steps=num_sim_steps,
                reward=reward,
                done=done,
                info=info,
                metadata={"plan_reasons": plan_reasons},
            )

        if artifact_dir is not None and num_sim_steps > 0 and hasattr(self.env, "export_render_to_video"):
            self.env.export_render_to_video(
                os.path.join(artifact_dir, "execute.{}".format(self.video_format)),
                out_type=self.video_format,
                fps=50,
            )

        return SkillExecutionResult(
            success=True,
            status=SkillExecutionStatus.SUCCESS,
            reason="",
            num_sim_steps=num_sim_steps,
            reward=reward,
            done=done,
            info=info,
            metadata={"plan_reasons": plan_reasons},
        )
