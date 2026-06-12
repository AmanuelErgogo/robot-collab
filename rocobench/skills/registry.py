from typing import Dict, List, Optional

from .models import (
    SkillCall,
    SkillSpec,
    SkillValidationResult,
    ValidationIssue,
    normalize_skill_name,
)


class SkillRegistry:
    """Deterministic registry for planner-visible skill declarations."""

    def __init__(self) -> None:
        self._skills = {}  # type: Dict[str, SkillSpec]
        self._aliases = {}  # type: Dict[str, str]

    def register(self, spec: SkillSpec) -> None:
        """Register a skill spec and its aliases."""
        canonical = normalize_skill_name(spec.name)
        if canonical in self._skills:
            raise ValueError("Duplicate skill name: {}".format(canonical))
        if canonical in self._aliases:
            raise ValueError("Skill name collides with alias: {}".format(canonical))

        for alias in spec.aliases:
            normalized_alias = normalize_skill_name(alias)
            if normalized_alias in self._skills or normalized_alias in self._aliases:
                raise ValueError("Skill alias collides with an existing name: {}".format(alias))

        self._skills[canonical] = spec
        for alias in spec.aliases:
            self._aliases[normalize_skill_name(alias)] = canonical

    def resolve_name(self, name_or_alias: str) -> Optional[str]:
        """Resolve a name or alias to a canonical skill name."""
        normalized = normalize_skill_name(name_or_alias)
        if normalized in self._skills:
            return normalized
        return self._aliases.get(normalized)

    def get(self, name_or_alias: str) -> SkillSpec:
        """Return the skill spec for a canonical name or alias."""
        canonical = self.resolve_name(name_or_alias)
        if canonical is None:
            raise KeyError("Unknown skill: {}".format(name_or_alias))
        return self._skills[canonical]

    def has(self, name_or_alias: str) -> bool:
        """Return whether the registry contains a skill name or alias."""
        return self.resolve_name(name_or_alias) is not None

    def skills_for_agent(self, agent_name: str) -> List[SkillSpec]:
        """Return all skills available to an agent.

        A spec with ``supported_agents=()`` is treated as available to all
        configured agents.
        """
        specs = []
        for name in sorted(self._skills):
            spec = self._skills[name]
            if not spec.supported_agents or agent_name in spec.supported_agents:
                specs.append(spec)
        return specs

    def canonicalize_call(self, call: SkillCall) -> SkillCall:
        """Return a call with canonical skill name and spec argument order."""
        spec = self.get(call.skill_name)
        ordered = {}
        for arg_name in spec.argument_names:
            if arg_name in call.arguments:
                ordered[arg_name] = call.arguments[arg_name]
        for arg_name in sorted(call.arguments):
            if arg_name not in ordered:
                ordered[arg_name] = call.arguments[arg_name]
        return SkillCall(
            agent_name=call.agent_name,
            skill_name=spec.name,
            arguments=ordered,
            raw_action=call.raw_action,
        )

    def validate_call_shape(self, call: SkillCall) -> SkillValidationResult:
        """Validate skill existence, agent support, and argument names."""
        issues = []
        canonical = self.resolve_name(call.skill_name)
        if canonical is None:
            return SkillValidationResult.invalid([
                ValidationIssue(
                    code="UNKNOWN_SKILL",
                    message="{} cannot execute unknown skill {}.".format(call.agent_name, call.skill_name),
                    agent_name=call.agent_name,
                )
            ])

        spec = self._skills[canonical]
        if spec.supported_agents and call.agent_name not in spec.supported_agents:
            issues.append(ValidationIssue(
                code="UNSUPPORTED_AGENT",
                message="{} cannot execute {}.".format(call.agent_name, spec.name),
                agent_name=call.agent_name,
            ))

        arg_names = set(call.arguments.keys())
        required = set(spec.required_arguments)
        allowed = set(spec.argument_names)
        for missing in sorted(required - arg_names):
            issues.append(ValidationIssue(
                code="MISSING_ARGUMENT",
                message="{} requires argument {}.".format(spec.name, missing),
                agent_name=call.agent_name,
            ))
        for unknown in sorted(arg_names - allowed):
            issues.append(ValidationIssue(
                code="UNKNOWN_ARGUMENT",
                message="{} does not accept argument {}.".format(spec.name, unknown),
                agent_name=call.agent_name,
            ))

        if issues:
            return SkillValidationResult.invalid(issues)
        return SkillValidationResult.ok()

    def render_agent_skill_prompt(self, agent_name: str) -> str:
        """Render available skills for one agent in deterministic order."""
        lines = []
        for index, spec in enumerate(self.skills_for_agent(agent_name), start=1):
            if spec.argument_names:
                args = ", ".join("{}=<{}>".format(name, name) for name in spec.argument_names)
            else:
                args = ""
            lines.append("{}. {}({})".format(index, spec.name, args))
            lines.append("   {}".format(spec.description))
        return "\n".join(lines)
