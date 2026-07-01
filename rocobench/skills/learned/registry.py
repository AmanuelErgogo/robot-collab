"""Deterministic learned policy registry."""

import os
from typing import Dict, Iterable, List, Optional

from rocobench.skills.models import normalize_skill_name

from .errors import PolicyRegistryError
from .models import LearnedPolicySpec


class LearnedPolicyRegistry(object):
    def __init__(self, specs=None):
        self._specs = {}  # type: Dict[str, LearnedPolicySpec]
        for spec in specs or ():
            self.register(spec)

    def register(self, spec):
        if spec.policy_id in self._specs:
            raise PolicyRegistryError("DUPLICATE_POLICY", "Duplicate learned policy_id: {}".format(spec.policy_id))
        self._specs[spec.policy_id] = spec

    def all_specs(self):
        return [self._specs[key] for key in sorted(self._specs)]

    def get(self, policy_id):
        try:
            return self._specs[str(policy_id)]
        except KeyError:
            raise PolicyRegistryError("POLICY_NOT_FOUND", "Unknown learned policy_id: {}".format(policy_id))

    def resolve(self, skill_name, agent_name, embodiment_id, task_id):
        canonical_skill = normalize_skill_name(skill_name)
        matches = []
        for spec in self.all_specs():
            if not spec.enabled:
                continue
            if normalize_skill_name(spec.skill_name) != canonical_skill:
                continue
            if spec.agent_name != agent_name:
                continue
            if spec.embodiment_id != embodiment_id:
                continue
            if spec.task_id != task_id:
                continue
            matches.append(spec)
        if not matches:
            raise PolicyRegistryError(
                "POLICY_NOT_FOUND",
                "No learned policy for skill={}, agent={}, embodiment={}, task={}.".format(
                    canonical_skill, agent_name, embodiment_id, task_id
                ),
            )
        if len(matches) > 1:
            raise PolicyRegistryError(
                "AMBIGUOUS_POLICY",
                "Ambiguous learned policy resolution.",
                evidence={"policy_ids": [spec.policy_id for spec in matches]},
            )
        return matches[0]

    _VALID_POLICY_TYPES = {"act", "octo", "mock", "lerobot", "bc_nn"}

    def validate_static(self, spec):
        issues = []
        if spec.policy_type.lower() not in self._VALID_POLICY_TYPES:
            issues.append(f"policy_type must be one of {sorted(self._VALID_POLICY_TYPES)}")
        if spec.execution_horizon > spec.max_steps:
            issues.append("execution_horizon cannot exceed max_steps")
        if spec.checkpoint and not os.path.exists(spec.checkpoint):
            issues.append("checkpoint path does not exist: {}".format(spec.checkpoint))
        if issues:
            raise PolicyRegistryError(
                "POLICY_SPEC_INVALID",
                "Invalid learned policy spec.",
                evidence={"issues": issues, "policy_id": spec.policy_id},
            )
        return True

