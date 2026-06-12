import logging
from typing import Optional, Tuple

from rocobench.skills.compiler import SkillCompilationError, SkillCompiler
from rocobench.skills.models import SkillPlan
from rocobench.skills.validation import SkillPlanValidator


LOGGER = logging.getLogger(__name__)


class SkillFeedbackManager:
    """Adapter with the legacy ``give_feedback(plan)`` shape for skill plans."""

    def __init__(self, validator: SkillPlanValidator, compiler: SkillCompiler, geometric_feedback_manager) -> None:
        self.validator = validator
        self.compiler = compiler
        self.geometric_feedback_manager = geometric_feedback_manager
        self._obs = None

    def update_obs(self, obs) -> None:
        """Set the observation used for validation and compilation."""
        self._obs = obs

    def give_feedback(self, skill_plan: SkillPlan) -> Tuple[bool, str]:
        """Prepare a skill plan or return actionable replanning feedback."""
        skill_plan.prepared_execution = None
        if self._obs is None:
            return False, "Skill plan could not be prepared:\n[NO_OBSERVATION] No observation is available for validation."

        validation = self.validator.validate(skill_plan, self._obs)
        if not validation.valid:
            return False, "Skill plan could not be prepared:\n{}".format(validation.to_feedback())

        try:
            prepared = self.compiler.compile(skill_plan, self._obs)
        except SkillCompilationError as exc:
            return False, "Skill plan could not be compiled:\n[{}] {}".format(exc.code, exc.message)
        except Exception as exc:
            LOGGER.exception("Unexpected skill compilation failure")
            return False, "Skill plan could not be compiled:\n[COMPILATION_FAILED] {}".format(str(exc))

        for compiled_plan in prepared.compiled_plans:
            ready, feedback = self.geometric_feedback_manager.give_feedback(compiled_plan)
            if not ready:
                return False, "Skill plan could not be prepared by the RRT backend:\n{}".format(feedback)

        skill_plan.prepared_execution = prepared
        return True, "[Environment Feedback]:\nSkill plan prepared successfully."
