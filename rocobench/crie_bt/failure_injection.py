"""Failure scenario helpers for scripted CRIE-BT experiments."""

from dataclasses import dataclass
from typing import Dict

from .executor import ScriptedSkillExecutor


SUPPORTED_SCENARIOS = (
    "none",
    "missed_grasp",
    "slippage",
    "no_progress",
    "target_occupied",
    "human_interrupt",
)


@dataclass(frozen=True)
class FailureInjectionConfig:
    scenario: str = "none"
    fail_attempts: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {"scenario": self.scenario, "fail_attempts": int(self.fail_attempts)}


def build_scripted_executor_for_scenario(config: FailureInjectionConfig) -> ScriptedSkillExecutor:
    scenario = config.scenario
    if scenario not in SUPPORTED_SCENARIOS:
        scenario = "none"
    return ScriptedSkillExecutor(scenario=scenario, fail_attempts=config.fail_attempts)
