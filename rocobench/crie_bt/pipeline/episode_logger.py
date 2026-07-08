"""Episode + event logging for the condition matrix.

Every episode row carries the full condition metadata required by
``docs/crie_next_stage_plan/logging_schema.json`` so downstream analysis always knows
which condition produced it.  Per-episode ``events.jsonl`` captures planner
calls, skill feedback, monitor updates, replans, and human interaction.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .conditions import ConditionConfig

# Counters that appear on every episode row.
_COUNTERS = (
    "num_steps",
    "planner_calls",
    "replans",
    "local_retries",
    "failed_subtasks",
    "dialogue_turns",
    "human_interventions",
    "monitor_updates",
)

REQUIRED_ROW_FIELDS = (
    "episode_id",
    "stage",
    "condition_name",
    "controller_family",
    "team_type",
    "coordination_mode",
    "skill_backend",
    "monitor_backend",
    "monitor_privileged",
    "environment",
    "planner_input_type",
    "task_id",
    "seed",
    "episode_index",
    "success",
    "task_done",
    "planner_calls",
    "replans",
    "local_retries",
    "failed_subtasks",
    "dialogue_turns",
    "human_interventions",
    "monitor_updates",
    "wall_time_s",
)

EVENT_TYPES = (
    "episode_start",
    "planner_call_start",
    "planner_call_end",
    "bt_tick",
    "skill_start",
    "skill_feedback",
    "skill_success",
    "skill_failure",
    "monitor_update",
    "replan_triggered",
    "local_retry",
    "human_instruction",
    "human_response",
    "dialogue_message",
    "episode_end",
)


class EpisodeLogger:
    def __init__(
        self,
        condition: ConditionConfig,
        task_id: str,
        seed: int,
        episode_index: int,
        events_path: Optional[str] = None,
    ) -> None:
        self.condition = condition
        self.task_id = task_id
        self.seed = int(seed)
        self.episode_index = int(episode_index)
        self.episode_id = "{stage}_{task}_seed{seed}_ep{idx:03d}_{code}".format(
            stage=condition.stage,
            task=task_id,
            seed=seed,
            idx=episode_index,
            code=condition.code_name,
        )
        self.counters: Dict[str, int] = {name: 0 for name in _COUNTERS}
        self.events: List[Dict[str, Any]] = []
        self._events_path = events_path
        self._events_handle = None
        if events_path:
            os.makedirs(os.path.dirname(os.path.abspath(events_path)), exist_ok=True)
            self._events_handle = open(events_path, "w", encoding="utf-8")
        self._start_time = time.perf_counter()

    # -- counters --------------------------------------------------------------
    def incr(self, name: str, amount: int = 1) -> None:
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += int(amount)

    # -- events ----------------------------------------------------------------
    def event(self, event_type: str, **payload: Any) -> None:
        row = {
            "timestamp": round(time.perf_counter() - self._start_time, 6),
            "episode_id": self.episode_id,
            "event_type": event_type,
            "condition_name": self.condition.condition_name,
        }
        row.update(payload)
        self.events.append(row)
        if self._events_handle is not None:
            self._events_handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            self._events_handle.flush()

    # -- finalize --------------------------------------------------------------
    def build_row(
        self,
        success: bool,
        task_done: bool,
        wall_time_s: Optional[float] = None,
        completion_time_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if wall_time_s is None:
            wall_time_s = time.perf_counter() - self._start_time
        row: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "stage": self.condition.stage,
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "success": bool(success),
            "task_done": bool(task_done),
            "completion_time_s": completion_time_s,
            "wall_time_s": round(float(wall_time_s), 6),
        }
        row.update(self.condition.metadata())
        row.update({name: int(value) for name, value in self.counters.items()})
        if extra:
            row.update(extra)
        return row

    def close(self) -> None:
        if self._events_handle is not None:
            self._events_handle.close()
            self._events_handle = None


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


__all__ = ["EVENT_TYPES", "EpisodeLogger", "REQUIRED_ROW_FIELDS"]
