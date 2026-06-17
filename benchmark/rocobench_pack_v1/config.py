"""Configuration for Phase 8 benchmark evaluation."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from .version import SUITE_NAME


TRACK_SKILL_POLICY = "skill_policy"
TRACK_PLANNER_SKILL = "planner_skill"
TRACKS = (TRACK_SKILL_POLICY, TRACK_PLANNER_SKILL)

METHOD_RRT = "rrt"
METHOD_HOLD = "hold"
METHOD_RANDOM = "random"
METHOD_ACT = "act"
METHOD_LEARNED_RRT_FALLBACK = "learned_rrt_fallback"
METHOD_EXTERNAL = "external"
METHODS = (METHOD_RRT, METHOD_HOLD, METHOD_RANDOM, METHOD_ACT, METHOD_LEARNED_RRT_FALLBACK, METHOD_EXTERNAL)

DEFAULT_TASKS_BY_TRACK = {
    TRACK_SKILL_POLICY: ("pack.put.alice", "pack.put.bob"),
    TRACK_PLANNER_SKILL: ("pack.sequential.two_agent", "pack.concurrent.safe_two_agent"),
}


@dataclass(frozen=True)
class EvaluationRunConfig:
    suite: str = SUITE_NAME
    track: str = TRACK_SKILL_POLICY
    method: str = METHOD_HOLD
    output_dir: str = "artifacts/benchmark/run_001"
    episodes_per_task: int = 2
    task_ids: Tuple[str, ...] = ()
    policy_path: Optional[str] = None
    endpoint: str = "tcp://127.0.0.1:5557"
    request_timeout_ms: int = 30000
    execution_horizon: Optional[int] = None
    action_bound_tolerance: float = 1e-5
    record_video: bool = False
    render_every_steps: int = 1
    overwrite: bool = False
    require_bridge: bool = True
    require_lerobot: bool = True
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.suite != SUITE_NAME:
            raise ValueError("unsupported suite: {}".format(self.suite))
        if self.track not in TRACKS:
            raise ValueError("unsupported track: {}".format(self.track))
        if self.method not in METHODS:
            raise ValueError("unsupported method: {}".format(self.method))
        if int(self.episodes_per_task) <= 0:
            raise ValueError("episodes_per_task must be positive")
        object.__setattr__(self, "task_ids", tuple(str(x) for x in self.task_ids))
        object.__setattr__(self, "notes", tuple(str(x) for x in self.notes))

    @property
    def resolved_task_ids(self) -> Tuple[str, ...]:
        return self.task_ids or DEFAULT_TASKS_BY_TRACK[self.track]

    def with_overrides(self, **kwargs: Any) -> "EvaluationRunConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["task_ids"] = list(self.task_ids)
        data["notes"] = list(self.notes)
        data["resolved_task_ids"] = list(self.resolved_task_ids)
        return data

