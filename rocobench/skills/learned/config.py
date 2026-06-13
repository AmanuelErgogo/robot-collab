"""Configuration helpers for learned skill execution."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .fallback import FallbackConfig
from .models import LearnedPolicySpec


@dataclass(frozen=True)
class LearnedExecutorConfig:
    task_id: str = "pack"
    embodiment_id: str = "ur5e_robotiq"
    max_active_calls: int = 1
    stable_success_checks: int = 2
    no_progress_window: int = 20
    no_progress_patience: int = 3
    no_progress_state_epsilon: float = 1e-4
    no_progress_action_epsilon: float = 1e-4
    action_bound_tolerance: float = 1e-5
    policy_cache_size: int = 1
    fallback: FallbackConfig = FallbackConfig()

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "embodiment_id": self.embodiment_id,
            "max_active_calls": int(self.max_active_calls),
            "stable_success_checks": int(self.stable_success_checks),
            "no_progress_window": int(self.no_progress_window),
            "no_progress_patience": int(self.no_progress_patience),
            "no_progress_state_epsilon": float(self.no_progress_state_epsilon),
            "no_progress_action_epsilon": float(self.no_progress_action_epsilon),
            "action_bound_tolerance": float(self.action_bound_tolerance),
            "policy_cache_size": int(self.policy_cache_size),
            "fallback": self.fallback.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        fallback = FallbackConfig.from_dict(data.get("fallback", {}))
        return cls(
            task_id=str(data.get("task_id", "pack")),
            embodiment_id=str(data.get("embodiment_id", "ur5e_robotiq")),
            max_active_calls=int(data.get("max_active_calls", 1)),
            stable_success_checks=int(data.get("stable_success_checks", 2)),
            no_progress_window=int(data.get("no_progress_window", 20)),
            no_progress_patience=int(data.get("no_progress_patience", 3)),
            no_progress_state_epsilon=float(data.get("no_progress_state_epsilon", 1e-4)),
            no_progress_action_epsilon=float(data.get("no_progress_action_epsilon", 1e-4)),
            action_bound_tolerance=float(data.get("action_bound_tolerance", 1e-5)),
            policy_cache_size=int(data.get("policy_cache_size", 1)),
            fallback=fallback,
        )


def load_learned_skill_config(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    data = dict(data or {})
    executor = LearnedExecutorConfig.from_dict(data.get("executor", {}))
    policies = [LearnedPolicySpec.from_dict(item) for item in data.get("policies", ())]
    return executor, policies

