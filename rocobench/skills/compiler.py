from typing import Any, Dict, List, Optional

from .models import PreparedSkillExecution, SkillPlan
from .pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT


class SkillCompilationError(RuntimeError):
    """Raised when a skill plan cannot be compiled for a backend."""

    def __init__(self, code: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        super(SkillCompilationError, self).__init__(message)
        self.code = code
        self.message = message
        self.metadata = metadata or {}


class SkillCompiler:
    """Base interface for converting skills to a backend executable plan."""

    def compile(self, plan: SkillPlan, obs) -> PreparedSkillExecution:
        raise NotImplementedError


class RRTSkillCompiler(SkillCompiler):
    """Compile grocery skill plans into legacy action-only RoCo path plans."""

    def __init__(self, env, legacy_parser) -> None:
        self.env = env
        self.legacy_parser = legacy_parser

    def compile(self, plan: SkillPlan, obs) -> PreparedSkillExecution:
        synthetic_response = self.build_synthetic_response(plan, obs)
        success, message, compiled_plans = self.legacy_parser.parse(obs, synthetic_response)
        if not success:
            raise SkillCompilationError(
                code="COMPILATION_FAILED",
                message=message,
                metadata={"synthetic_response": synthetic_response},
            )
        return PreparedSkillExecution(
            backend_name="rrt",
            source_plan_id=plan.plan_id,
            compiled_plans=list(compiled_plans),
            metadata={"synthetic_response": synthetic_response},
        )

    def build_synthetic_response(self, plan: SkillPlan, obs) -> str:
        """Return the deterministic legacy action-only response for a skill plan."""
        lines = ["EXECUTE"]
        for call in plan.calls:
            if call.skill_name == WAIT:
                legacy_action = "WAIT"
            elif call.skill_name == PUT_OBJECT_IN_CONTAINER:
                obj_name = call.arguments["object"]
                container = call.arguments["container"]
                held_obj = self._get_agent_held_object(obs, call.agent_name)
                if held_obj == obj_name:
                    legacy_action = "PLACE {} {}".format(obj_name, container)
                else:
                    legacy_action = "PICK {} PLACE {}".format(obj_name, container)
            else:
                raise SkillCompilationError(
                    code="UNKNOWN_SKILL",
                    message="Cannot compile unsupported skill {}.".format(call.skill_name),
                )
            lines.append("NAME {} ACTION {}".format(call.agent_name, legacy_action))
        return "\n".join(lines)

    def _get_agent_held_object(self, obs, agent_name: str) -> Optional[str]:
        if hasattr(self.env, "get_agent_held_object"):
            return self.env.get_agent_held_object(obs, agent_name)
        robot_name = getattr(self.env, "robot_name_map_inv", {}).get(agent_name)
        robot_state = getattr(obs, robot_name, None) if robot_name is not None else None
        contacts = getattr(robot_state, "contacts", set()) if robot_state is not None else set()
        for item_name in getattr(self.env, "item_names", []):
            if item_name in contacts:
                return item_name
        return None
