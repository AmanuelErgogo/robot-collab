"""Condition registry: the single source of truth for the eight paper conditions.

Every run resolves a ``(condition_name, stage)`` pair into a fully specified
:class:`ConditionConfig` via :func:`build_condition`.  This replaces the ad hoc
``mode`` / ``planner_mode`` combinations used by the legacy runner and guarantees
that each episode row records exactly which condition it executed.

See ``docs/crie_next_stage_plan/02_condition_taxonomy.md`` and
``docs/crie_next_stage_plan/condition_registry.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .interfaces import (
    ControllerFamily,
    CoordinationMode,
    EnvironmentType,
    MonitorBackend,
    SkillBackend,
    TeamType,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_REGISTRY_PATH = os.path.join(_REPO_ROOT, "configs", "conditions", "condition_registry.yaml")

VALID_STAGES = ("step1", "step2", "step2a", "step3")


@dataclass(frozen=True)
class ConditionConfig:
    """Fully resolved specification for one condition at one stage."""

    condition_name: str
    code_name: str
    controller_family: ControllerFamily
    team_type: TeamType
    coordination_mode: CoordinationMode
    baseline: bool
    explicit_monitor: bool
    skill_backend: SkillBackend
    monitor_backend: MonitorBackend
    monitor_privileged: bool
    environment: EnvironmentType
    planner_input_type: str
    stage: str
    tasks: List[str] = field(default_factory=list)
    human_interface: Optional[str] = None
    description: str = ""

    def metadata(self) -> Dict[str, Any]:
        """The condition metadata that must appear on every episode row."""
        return {
            "condition_name": self.condition_name,
            "controller_family": self.controller_family,
            "team_type": self.team_type,
            "coordination_mode": self.coordination_mode,
            "skill_backend": self.skill_backend,
            "monitor_backend": self.monitor_backend,
            "monitor_privileged": self.monitor_privileged,
            "environment": self.environment,
            "planner_input_type": self.planner_input_type,
        }


class ConditionRegistry:
    """Loads and resolves conditions from ``condition_registry.yaml``."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._conditions: Dict[str, Any] = dict(data.get("conditions", {}))
        self._stages: Dict[str, Any] = dict(data.get("stages", {}))
        if not self._conditions:
            raise ValueError("Condition registry has no 'conditions' section.")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ConditionRegistry":
        path = path or DEFAULT_REGISTRY_PATH
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(data or {})

    def condition_names(self) -> List[str]:
        return list(self._conditions.keys())

    def code_name_map(self) -> Dict[str, str]:
        return {name: spec["code_name"] for name, spec in self._conditions.items()}

    def resolve_name(self, name: str) -> str:
        """Accept either a paper name (``CRIE-BT-RR-Cent``) or a code name."""
        if name in self._conditions:
            return name
        for paper_name, spec in self._conditions.items():
            if spec.get("code_name") == name:
                return paper_name
        raise KeyError("Unknown condition: {!r}. Known: {}".format(name, self.condition_names()))

    def stage_spec(self, stage: str) -> Dict[str, Any]:
        # step2a is an alias of step2 (terminal human-robot simulation).
        lookup = "step2" if stage == "step2a" else stage
        if lookup not in self._stages:
            raise KeyError("Unknown stage: {!r}. Known: {}".format(stage, list(self._stages.keys())))
        return dict(self._stages[lookup])

    def build_condition(self, condition_name: str, stage: str) -> ConditionConfig:
        if stage not in VALID_STAGES:
            raise ValueError("Unknown stage {!r}; expected one of {}".format(stage, VALID_STAGES))
        paper_name = self.resolve_name(condition_name)
        spec = self._conditions[paper_name]
        stage_spec = self.stage_spec(stage)
        stage_key = "step2" if stage == "step2a" else stage

        baseline = bool(spec.get("baseline", False))
        # The monitor backend is a stage property, but the baseline never runs an
        # explicit monitor: it self-monitors inside the VLM planning loop.
        if baseline:
            monitor_backend: MonitorBackend = stage_spec.get("baseline_monitor_backend", "VLM-self")
            monitor_privileged = False
        else:
            monitor_backend = spec.get("default_monitor_backend", {}).get(
                stage_key, stage_spec.get("criebt_monitor_backend", "CodedSim")
            )
            monitor_privileged = monitor_backend == "CodedSim"

        return ConditionConfig(
            condition_name=paper_name,
            code_name=spec["code_name"],
            controller_family=spec["controller_family"],
            team_type=spec["team_type"],
            coordination_mode=spec["coordination_mode"],
            baseline=baseline,
            explicit_monitor=bool(spec.get("explicit_monitor", not baseline)),
            skill_backend=stage_spec.get("skill_backend", "RRT"),
            monitor_backend=monitor_backend,
            monitor_privileged=monitor_privileged,
            environment=stage_spec.get("environment", "sim"),
            planner_input_type=spec.get("planner_input_type", "image_status_history"),
            stage=stage,
            tasks=list(stage_spec.get("tasks", [])),
            human_interface=stage_spec.get("human_interface"),
            description=spec.get("description", ""),
        )


_DEFAULT_REGISTRY: Optional[ConditionRegistry] = None


def get_registry(path: Optional[str] = None) -> ConditionRegistry:
    """Return a cached registry (or a fresh one when ``path`` is given)."""
    global _DEFAULT_REGISTRY
    if path is not None:
        return ConditionRegistry.load(path)
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ConditionRegistry.load()
    return _DEFAULT_REGISTRY


def build_condition(condition_name: str, stage: str, registry_path: Optional[str] = None) -> ConditionConfig:
    """Resolve a condition name + stage into a fully specified config."""
    return get_registry(registry_path).build_condition(condition_name, stage)


def all_condition_names(registry_path: Optional[str] = None) -> List[str]:
    return get_registry(registry_path).condition_names()


__all__ = [
    "ConditionConfig",
    "ConditionRegistry",
    "DEFAULT_REGISTRY_PATH",
    "VALID_STAGES",
    "all_condition_names",
    "build_condition",
    "get_registry",
]
