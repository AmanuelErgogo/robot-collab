"""Static metadata registry: tasks, agents, embodiments, skills, recipes."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Task / agent / skill metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentMeta:
    name: str
    embodiment_id: str
    description: str = ""


@dataclass(frozen=True)
class TaskMeta:
    task_id: str
    class_path: str          # "module:ClassName"
    description: str
    agents: Tuple[AgentMeta, ...]
    skills: Tuple[str, ...]
    recipes: Tuple[str, ...] = ()   # named assembly sequences (sandwich only)

    @property
    def agent_names(self) -> List[str]:
        return [a.name for a in self.agents]

    @property
    def embodiment_ids(self) -> List[str]:
        return [a.embodiment_id for a in self.agents]


TASK_REGISTRY: Dict[str, TaskMeta] = {
    "sandwich": TaskMeta(
        task_id="sandwich",
        class_path="rocobench.envs.task_sandwich:MakeSandwichTask",
        description="Two-arm sandwich assembly: stack ingredients on a cutting board in recipe order.",
        agents=(
            AgentMeta("Chad", "ur5e_suction",  "UR5e with suction gripper"),
            AgentMeta("Dave", "humanoid",       "Humanoid torso"),
        ),
        skills=("PICK", "STACK_ON", "WAIT"),
        recipes=("bacon", "vegetarian", "beef_patty", "ham"),
    ),
    "pack": TaskMeta(
        task_id="pack",
        class_path="rocobench.envs.task_pack:PackGroceryTask",
        description="Pack groceries from a table into a container using two arms.",
        agents=(
            AgentMeta("Alice", "ur5e_robotiq", "UR5e with Robotiq gripper"),
            AgentMeta("Bob",   "panda",         "Franka Panda"),
        ),
        skills=("PICK", "PLACE", "PICK_AND_PLACE", "WAIT"),
    ),
    "cabinet": TaskMeta(
        task_id="cabinet",
        class_path="rocobench.envs.task_cabinet:OpenCabinetTask",
        description="Open a cabinet door then pick and place an object inside.",
        agents=(
            AgentMeta("Alice", "ur5e_robotiq", "UR5e with Robotiq gripper"),
            AgentMeta("Bob",   "panda",         "Franka Panda"),
        ),
        skills=("PICK", "PLACE", "OPEN_CABINET", "WAIT"),
    ),
    "sort": TaskMeta(
        task_id="sort",
        class_path="rocobench.envs.task_sort:SortTask",
        description="Sort coloured blocks into matching bins.",
        agents=(
            AgentMeta("Alice", "ur5e_robotiq", "UR5e with Robotiq gripper"),
            AgentMeta("Bob",   "panda",         "Franka Panda"),
        ),
        skills=("PICK", "PLACE", "WAIT"),
    ),
    "rope": TaskMeta(
        task_id="rope",
        class_path="rocobench.envs.task_rope:RopeTask",
        description="Collaborative rope manipulation to reach a target configuration.",
        agents=(
            AgentMeta("Alice", "ur5e_robotiq", "UR5e with Robotiq gripper"),
            AgentMeta("Bob",   "panda",         "Franka Panda"),
        ),
        skills=("PICK", "PLACE", "WAIT"),
    ),
    "sweep": TaskMeta(
        task_id="sweep",
        class_path="rocobench.envs.task_sweep:SweepTask",
        description="Sweep debris into a dustpan with a broom.",
        agents=(
            AgentMeta("Alice", "ur5e_robotiq", "UR5e with Robotiq gripper"),
        ),
        skills=("SWEEP", "WAIT"),
    ),
}

DEFAULT_SKILLS_PER_TASK: Dict[str, List[str]] = {
    t.task_id: [s for s in t.skills if s != "WAIT"]
    for t in TASK_REGISTRY.values()
}

SANDWICH_RECIPES: Dict[str, List[str]] = {
    "bacon":      ["bread_slice1", "bacon",      "cheese", "tomato",            "bread_slice2"],
    "vegetarian": ["bread_slice1", "cheese",     "tomato", "cucumber",          "bread_slice2"],
    "beef_patty": ["bread_slice1", "beef_patty", "cheese", "tomato",            "bread_slice2"],
    "ham":        ["bread_slice1", "ham",        "cheese", "tomato", "cucumber","bread_slice2"],
}


# ---------------------------------------------------------------------------
# Trained-skill inventory (reads configs/skills/*.yaml)
# ---------------------------------------------------------------------------

@dataclass
class TrainedSkillEntry:
    policy_id: str
    skill_name: str
    agent_name: str
    embodiment_id: str
    task_id: str
    checkpoint: str
    policy_type: str
    enabled: bool
    config_file: str = ""
    execution_horizon: int = 10
    max_steps: int = 300
    schema_hash: str = ""
    action_representation: str = "joint_ctrl"
    cameras: tuple = ()
    failure_monitors: tuple = ("TIMEOUT", "NO_PROGRESS", "NONFINITE_ACTION")

    @property
    def checkpoint_exists(self) -> bool:
        if self.policy_type == "mock":
            return True
        return bool(self.checkpoint) and os.path.exists(self.checkpoint)


def _load_yaml_or_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".json") or yaml is None:
            return json.load(fh)
        return yaml.safe_load(fh) or {}


def scan_trained_skills(
    configs_dir: Optional[str] = None,
    task_filter: Optional[str] = None,
) -> List[TrainedSkillEntry]:
    """Scan configs/skills/*.yaml and return all registered policy specs."""
    if configs_dir is None:
        configs_dir = os.path.join(
            os.path.dirname(__file__), "..", "configs", "skills"
        )
    configs_dir = os.path.abspath(configs_dir)
    pattern = os.path.join(configs_dir, "*.yaml")
    entries: List[TrainedSkillEntry] = []
    for cfg_path in sorted(glob.glob(pattern)):
        try:
            data = _load_yaml_or_json(cfg_path)
        except Exception:
            continue
        for spec in data.get("policies", []):
            if not isinstance(spec, dict):
                continue
            fm = spec.get("failure_monitors", ["TIMEOUT", "NO_PROGRESS", "NONFINITE_ACTION"])
            entry = TrainedSkillEntry(
                policy_id=str(spec.get("policy_id", "")),
                skill_name=str(spec.get("skill_name", "")),
                agent_name=str(spec.get("agent_name", "")),
                embodiment_id=str(spec.get("embodiment_id", "")),
                task_id=str(spec.get("task_id", "")),
                checkpoint=str(spec.get("checkpoint", "")),
                policy_type=str(spec.get("policy_type", "unknown")),
                enabled=bool(spec.get("enabled", True)),
                config_file=cfg_path,
                execution_horizon=int(spec.get("execution_horizon", 10)),
                max_steps=int(spec.get("max_steps", 300)),
                schema_hash=str(spec.get("schema_hash", "")),
                action_representation=str(spec.get("action_representation", "joint_ctrl")),
                cameras=tuple(spec.get("cameras", [])),
                failure_monitors=tuple(fm),
            )
            if task_filter and entry.task_id != task_filter:
                continue
            entries.append(entry)
    return entries
