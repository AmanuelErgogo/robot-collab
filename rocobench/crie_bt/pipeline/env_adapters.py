"""Environment adapters.

:class:`SyntheticEnvironmentAdapter` is a lightweight, deterministic staged-task
environment used for dry-runs, the condition-matrix smoke test, and unit tests.
It exposes exactly the data split the fairness rule requires:

* ``public_percepts`` -- non-privileged task structure a planner may read;
* ``oracle_state`` -- privileged ground-truth stage progress that only the coded
  simulator monitor and the evaluator may read.

Real RoCoBench MuJoCo tasks plug in behind :class:`RoCoBenchEnvironmentAdapter`
(a documented extension point), reusing the same interface so controllers do not
change between Step 1 (sim) and Step 3 (real).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import EnvironmentAdapter, ObservationBundle, SkillCall


@dataclass
class StageSpec:
    stage_id: str
    agent_id: str
    skill_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    assignee: str = "robot"  # "robot" or "human"

    def descriptor(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "agent_id": self.agent_id,
            "skill_name": self.skill_name,
            "args": dict(self.args),
            "assignee": self.assignee,
        }


def _pick_place(stage_id: str, agent: str, obj: str, target: str, assignee: str = "robot") -> StageSpec:
    return StageSpec(stage_id, agent, "pick_place", {"object": obj, "target": target}, assignee)


# Synthetic staged tasks. Robot-robot (RR) tasks use robot_1/robot_2; human-robot
# (HR) tasks route some stages to human_1. Stage ids double as SARM stage ids.
SYNTHETIC_TASKS: Dict[str, List[StageSpec]] = {
    # ---- Step 1 robot-robot tasks -------------------------------------------
    "sandwich": [
        _pick_place("place_bread_bottom", "robot_1", "bread_bottom", "plate"),
        _pick_place("place_cheese", "robot_2", "cheese", "plate"),
        _pick_place("place_tomato", "robot_1", "tomato", "plate"),
        _pick_place("place_bread_top", "robot_2", "bread_top", "plate"),
    ],
    "pack": [
        _pick_place("pack_apple", "robot_1", "apple", "bin"),
        _pick_place("pack_banana", "robot_2", "banana", "bin"),
        _pick_place("pack_milk", "robot_1", "milk", "bin"),
    ],
    "cabinet": [
        StageSpec("open_cabinet", "robot_1", "open", {"target": "cabinet_door"}),
        _pick_place("stow_cup", "robot_2", "cup", "cabinet_shelf"),
        StageSpec("close_cabinet", "robot_1", "close", {"target": "cabinet_door"}),
    ],
    "sort": [
        _pick_place("sort_red_block", "robot_1", "red_block", "red_panel"),
        _pick_place("sort_blue_block", "robot_2", "blue_block", "blue_panel"),
    ],
    # ---- Step 2 human-robot simulation tasks --------------------------------
    "medication_sim": [
        _pick_place("grasp_medication_bottle", "robot_1", "red_bottle", "tray"),
        StageSpec("human_verify_dosage", "human_1", "verify", {"item": "red_bottle"}, assignee="human"),
        _pick_place("place_in_patient_box", "robot_1", "red_bottle", "patient_box"),
    ],
    "cooking_sim": [
        _pick_place("fetch_pan", "robot_1", "pan", "stove"),
        StageSpec("human_add_ingredient", "human_1", "add", {"item": "vegetables"}, assignee="human"),
        StageSpec("stir", "robot_1", "stir", {"target": "pan"}),
        _pick_place("plate_dish", "robot_1", "dish", "table"),
    ],
}


def available_synthetic_tasks() -> List[str]:
    return list(SYNTHETIC_TASKS.keys())


@dataclass
class SkillResult:
    status: str  # "success" | "failure"
    message: str
    stage_id: str
    matched: bool


class SyntheticEnvironmentAdapter(EnvironmentAdapter):
    """Deterministic staged-task environment for sim runs and tests."""

    environment = "sim"

    def __init__(
        self,
        fail_stage_ids: Optional[Dict[str, int]] = None,
    ) -> None:
        # fail_stage_ids maps a stage_id -> number of times it should fail before
        # succeeding, used to exercise retry/replan without external state.
        self._fail_budget: Dict[str, int] = dict(fail_stage_ids or {})
        self.task_id = ""
        self.stages: List[StageSpec] = []
        self.stage_index = 0
        self._fail_counts: Dict[str, int] = {}
        self._tick = 0

    # -- EnvironmentAdapter API ------------------------------------------------
    def reset(self, task_id: str, seed: int) -> ObservationBundle:
        if task_id not in SYNTHETIC_TASKS:
            raise KeyError("Unknown synthetic task {!r}. Known: {}".format(task_id, available_synthetic_tasks()))
        self.task_id = task_id
        self.stages = list(SYNTHETIC_TASKS[task_id])
        self.stage_index = 0
        self._fail_counts = {}
        self._tick = 0
        return self.observe()

    def observe(self) -> ObservationBundle:
        stage = self.current_stage()
        remaining = [s.descriptor() for s in self.stages[self.stage_index:]]
        public = {
            "task_id": self.task_id,
            "num_stages": len(self.stages),
            "completed_stages": self.stage_index,
            "next_stage": stage.descriptor() if stage is not None else None,
            "remaining_plan": remaining,
        }
        oracle = {
            "stage_id": stage.stage_id if stage is not None else "task_complete",
            "stage_index": self.stage_index,
            "stage_progress": 1.0 if stage is None else 0.0,
            "stage_done": stage is None,
            "task_done": self.get_task_done(),
        }
        return ObservationBundle(public_percepts=public, oracle_state=oracle, timestamp=float(self._tick))

    def step_robot(self, agent_id: str, low_level_action: Any) -> ObservationBundle:
        # In synthetic mode the "low-level action" is the SkillCall itself.
        if isinstance(low_level_action, SkillCall):
            self.execute_skill(low_level_action)
        return self.observe()

    def apply_human_action(self, human_id: str, action: Any) -> ObservationBundle:
        if isinstance(action, SkillCall):
            self.execute_skill(action)
        return self.observe()

    def get_task_done(self) -> Optional[bool]:
        return self.stage_index >= len(self.stages)

    def get_oracle_state(self) -> Dict[str, Any]:
        return dict(self.observe().oracle_state or {})

    # -- Synthetic-specific helpers used by SyntheticSkillExecutor -------------
    def current_stage(self) -> Optional[StageSpec]:
        if 0 <= self.stage_index < len(self.stages):
            return self.stages[self.stage_index]
        return None

    def execute_skill(self, skill_call: SkillCall) -> SkillResult:
        """Attempt ``skill_call`` against the current stage; advance on success."""
        self._tick += 1
        stage = self.current_stage()
        if stage is None:
            return SkillResult("failure", "Task already complete.", "task_complete", matched=False)

        matched = skill_call.agent_id == stage.agent_id and skill_call.skill_name == stage.skill_name
        if not matched:
            return SkillResult(
                "failure",
                "Skill {}({}) does not match expected stage {} for {}.".format(
                    skill_call.skill_name, skill_call.agent_id, stage.stage_id, stage.agent_id
                ),
                stage.stage_id,
                matched=False,
            )

        budget = self._fail_budget.get(stage.stage_id, 0)
        seen = self._fail_counts.get(stage.stage_id, 0)
        if seen < budget:
            self._fail_counts[stage.stage_id] = seen + 1
            return SkillResult("failure", "Injected failure on stage {}.".format(stage.stage_id),
                               stage.stage_id, matched=True)

        self.stage_index += 1
        return SkillResult("success", "Stage {} complete.".format(stage.stage_id), stage.stage_id, matched=True)


# The functional RoCoBench MuJoCo adapter lives in ``roco_backend`` so that
# importing this light module never pulls in MuJoCo/dm_control. See
# ``rocobench.crie_bt.pipeline.roco_backend.RoCoBenchEnvironmentAdapter``.


__all__ = [
    "SkillResult",
    "StageSpec",
    "SYNTHETIC_TASKS",
    "SyntheticEnvironmentAdapter",
    "available_synthetic_tasks",
]
