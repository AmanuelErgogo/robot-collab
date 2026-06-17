"""Versioned task declarations for RoCoBench-Pack-Skills-v1."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from .version import TASK_MANIFEST_PATH, read_json


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    description: str
    active_agents: Tuple[str, ...]
    max_steps: int
    success_predicate: str
    failure_predicates: Tuple[str, ...]
    action_representation: str
    required_cameras: Tuple[str, ...]
    variation_group: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkTask":
        return cls(
            task_id=str(data["task_id"]),
            description=str(data["description"]),
            active_agents=tuple(str(x) for x in data["active_agents"]),
            max_steps=int(data["max_steps"]),
            success_predicate=str(data["success_predicate"]),
            failure_predicates=tuple(str(x) for x in data["failure_predicates"]),
            action_representation=str(data["action_representation"]),
            required_cameras=tuple(str(x) for x in data["required_cameras"]),
            variation_group=str(data["variation_group"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "active_agents": list(self.active_agents),
            "max_steps": int(self.max_steps),
            "success_predicate": self.success_predicate,
            "failure_predicates": list(self.failure_predicates),
            "action_representation": self.action_representation,
            "required_cameras": list(self.required_cameras),
            "variation_group": self.variation_group,
        }


def load_task_manifest(path: str = TASK_MANIFEST_PATH) -> Dict[str, Any]:
    return read_json(path)


def load_tasks(path: str = TASK_MANIFEST_PATH) -> Tuple[BenchmarkTask, ...]:
    manifest = load_task_manifest(path)
    tasks = tuple(BenchmarkTask.from_dict(item) for item in manifest.get("tasks", ()))
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task_id in task manifest")
    return tasks


def tasks_by_id(tasks: Iterable[BenchmarkTask] = None) -> Dict[str, BenchmarkTask]:
    values = tuple(tasks) if tasks is not None else load_tasks()
    return {task.task_id: task for task in values}

