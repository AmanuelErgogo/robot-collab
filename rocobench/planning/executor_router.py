"""Deterministic skill executor routing for Phase 6."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from rocobench.skills.models import SkillCall, SkillPlan, normalize_skill_name
from rocobench.skills.pack_grocery import WAIT


BACKEND_LEARNED = "learned"
BACKEND_RRT = "rrt"
BACKEND_UNSUPPORTED = "unsupported"
SUPPORTED_BACKENDS = (BACKEND_LEARNED, BACKEND_RRT, BACKEND_UNSUPPORTED)


@dataclass(frozen=True)
class ExecutorRoute:
    """One configured route for an agent/skill pair.

    ``policy_id`` is internal metadata. Prompt renderers expose only capability
    availability and backend class, never checkpoint paths or policy IDs.
    """

    agent: str
    skill: str
    primary: str
    fallback: str = BACKEND_UNSUPPORTED
    policy_id: Optional[str] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        skill = "*" if self.skill == "*" else normalize_skill_name(self.skill)
        primary = str(self.primary).strip().lower()
        fallback = str(self.fallback).strip().lower()
        if primary not in SUPPORTED_BACKENDS:
            raise ValueError("Unsupported primary backend: {}".format(self.primary))
        if fallback not in SUPPORTED_BACKENDS:
            raise ValueError("Unsupported fallback backend: {}".format(self.fallback))
        object.__setattr__(self, "skill", skill)
        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "fallback", fallback)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutorRoute":
        data = dict(data)
        return cls(
            agent=str(data.get("agent", "*")),
            skill=str(data.get("skill", "*")),
            primary=str(data.get("primary", BACKEND_UNSUPPORTED)),
            fallback=str(data.get("fallback", BACKEND_UNSUPPORTED)),
            policy_id=data.get("policy_id"),
            enabled=bool(data.get("enabled", True)),
        )

    def matches(self, call: SkillCall) -> bool:
        if not self.enabled:
            return False
        agent_match = self.agent == "*" or self.agent == call.agent_name
        skill_match = self.skill == "*" or self.skill == normalize_skill_name(call.skill_name)
        return bool(agent_match and skill_match)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "skill": self.skill,
            "primary": self.primary,
            "fallback": self.fallback,
            "policy_id": self.policy_id,
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class ExecutorDecision:
    backend: str
    policy_id: Optional[str]
    reason: str
    fallback_backend: str = BACKEND_UNSUPPORTED
    route: Optional[ExecutorRoute] = None

    @property
    def supported(self) -> bool:
        return self.backend != BACKEND_UNSUPPORTED

    def to_dict(self, include_internal: bool = True) -> Dict[str, Any]:
        data = {
            "backend": self.backend,
            "reason": self.reason,
            "fallback_backend": self.fallback_backend,
        }
        if include_internal:
            data["policy_id"] = self.policy_id
            data["route"] = self.route.to_dict() if self.route is not None else None
        return data


class SkillExecutorRouter(object):
    """Resolve typed skill calls to internal executor backends."""

    def __init__(
        self,
        routes: Optional[Iterable[ExecutorRoute]] = None,
        default_backend: str = BACKEND_RRT,
        default_fallback: str = BACKEND_UNSUPPORTED,
    ) -> None:
        self.routes = list(routes or ())
        self.default_backend = str(default_backend).strip().lower()
        self.default_fallback = str(default_fallback).strip().lower()
        if self.default_backend not in SUPPORTED_BACKENDS:
            raise ValueError("Unsupported default backend: {}".format(default_backend))
        if self.default_fallback not in SUPPORTED_BACKENDS:
            raise ValueError("Unsupported default fallback: {}".format(default_fallback))

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "SkillExecutorRouter":
        data = dict(data or {})
        routes = [ExecutorRoute.from_dict(item) for item in data.get("routes", ())]
        return cls(
            routes=routes,
            default_backend=str(data.get("default_backend", BACKEND_RRT)),
            default_fallback=str(data.get("default_fallback", BACKEND_UNSUPPORTED)),
        )

    def resolve(self, call: SkillCall, context: Optional[Mapping[str, Any]] = None) -> ExecutorDecision:
        context = dict(context or {})
        if call.skill_name == WAIT:
            return ExecutorDecision(
                backend=BACKEND_RRT,
                policy_id=None,
                reason="WAIT is handled by the deterministic RRT/no-op backend.",
                fallback_backend=BACKEND_UNSUPPORTED,
            )
        if bool(context.get("disable_learned")):
            return ExecutorDecision(
                backend=BACKEND_RRT,
                policy_id=None,
                reason="Learned backend disabled by context.",
                fallback_backend=self.default_fallback,
            )

        for route in self.routes:
            if route.matches(call):
                return ExecutorDecision(
                    backend=route.primary,
                    policy_id=route.policy_id,
                    reason="Matched route for {} {}.".format(call.agent_name, call.skill_name),
                    fallback_backend=route.fallback,
                    route=route,
                )
        return ExecutorDecision(
            backend=self.default_backend,
            policy_id=None,
            reason="No explicit route matched; using default backend.",
            fallback_backend=self.default_fallback,
        )

    def select_active_call(self, plan: SkillPlan) -> Optional[SkillCall]:
        active = [call for call in plan.calls if call.skill_name != WAIT]
        if len(active) != 1:
            return None
        return active[0]

    def prompt_capabilities(self, agent_names: Sequence[str], registry: Any = None) -> str:
        """Render planner-visible capabilities without internal policy IDs."""
        lines = []
        for agent_name in agent_names:
            lines.append("{} capabilities:".format(agent_name))
            if registry is not None:
                specs = registry.skills_for_agent(agent_name)
            else:
                specs = []
            if not specs:
                lines.append("- Use configured typed skills only.")
            for spec in specs:
                route = self._best_route(agent_name, spec.name)
                backend = route.primary if route is not None else self.default_backend
                fallback = route.fallback if route is not None else self.default_fallback
                if backend == BACKEND_LEARNED:
                    detail = "learned capability available, not guaranteed"
                elif backend == BACKEND_RRT:
                    detail = "RRT capability available"
                else:
                    detail = "unsupported"
                if fallback != BACKEND_UNSUPPORTED:
                    detail += "; fallback: {}".format(fallback)
                lines.append("- {}: {}".format(spec.name, detail))
        return "\n".join(lines)

    def _best_route(self, agent_name: str, skill_name: str) -> Optional[ExecutorRoute]:
        probe = SkillCall(agent_name, skill_name, {}, "{}()".format(skill_name))
        for route in self.routes:
            if route.matches(probe):
                return route
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes": [route.to_dict() for route in self.routes],
            "default_backend": self.default_backend,
            "default_fallback": self.default_fallback,
        }


def active_call_or_issue(plan: SkillPlan) -> Optional[SkillCall]:
    active = [call for call in plan.calls if call.skill_name != WAIT]
    if len(active) != 1:
        return None
    return active[0]
