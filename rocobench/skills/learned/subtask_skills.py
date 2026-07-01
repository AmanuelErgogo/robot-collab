"""Subtask skill registries for all RoCo tasks.

Each task defines a small set of composable skills (PICK, PLACE, OPEN_CABINET,
STACK_ON, WAIT). These feed the LearnedPolicyRegistry so each skill can be
backed by a separate ACT checkpoint per robot embodiment.
"""

from typing import Sequence

from rocobench.skills.models import SkillSpec
from rocobench.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Canonical skill name constants
# ---------------------------------------------------------------------------
SKILL_PICK = "PICK"
SKILL_PLACE = "PLACE"
SKILL_PICK_AND_PLACE = "PICK_AND_PLACE"
SKILL_OPEN_CABINET = "OPEN_CABINET"
SKILL_STACK_ON = "STACK_ON"
SKILL_SWEEP = "SWEEP"
SKILL_WAIT = "WAIT"

ALL_SUBTASK_SKILLS = (
    SKILL_PICK,
    SKILL_PLACE,
    SKILL_PICK_AND_PLACE,
    SKILL_OPEN_CABINET,
    SKILL_STACK_ON,
    SKILL_SWEEP,
    SKILL_WAIT,
)


def build_pick_place_registry(agent_names: Sequence[str]) -> SkillRegistry:
    """Generic pick-and-place skills used by pack, sort, and rope tasks."""
    supported = tuple(agent_names)
    registry = SkillRegistry()
    registry.register(SkillSpec(
        name=SKILL_PICK,
        description=(
            "Approach the specified object, close the gripper around it, and lift it "
            "to a safe carry height. Fails if the object is already held or unreachable."
        ),
        required_arguments=("object",),
        supported_agents=supported,
        resource_arguments=("object",),
    ))
    registry.register(SkillSpec(
        name=SKILL_PLACE,
        description=(
            "Transport the currently-held object to the target location and release it. "
            "Fails if the robot is not holding anything or the target is occupied."
        ),
        required_arguments=("target",),
        supported_agents=supported,
        resource_arguments=("target",),
    ))
    registry.register(SkillSpec(
        name=SKILL_PICK_AND_PLACE,
        description=(
            "Atomically pick the specified object and place it at the target location. "
            "Equivalent to a sequential PICK then PLACE but executed as one policy."
        ),
        required_arguments=("object", "target"),
        supported_agents=supported,
        resource_arguments=("object", "target"),
    ))
    registry.register(SkillSpec(
        name=SKILL_WAIT,
        description="Hold position while the other agent executes its skill.",
        required_arguments=(),
        supported_agents=supported,
    ))
    return registry


def build_cabinet_registry(agent_names: Sequence[str]) -> SkillRegistry:
    """Skills for the cabinet task (pick, place, open door)."""
    registry = build_pick_place_registry(agent_names)
    supported = tuple(agent_names)
    registry.register(SkillSpec(
        name=SKILL_OPEN_CABINET,
        description=(
            "Grasp the specified door handle and swing the door to the open position. "
            "The robot must approach the handle before this skill can fire."
        ),
        required_arguments=("door",),
        supported_agents=supported,
        resource_arguments=("door",),
    ))
    return registry


def build_sandwich_registry(agent_names: Sequence[str]) -> SkillRegistry:
    """Skills for the sandwich assembly task."""
    supported = tuple(agent_names)
    registry = SkillRegistry()
    registry.register(SkillSpec(
        name=SKILL_PICK,
        description="Pick up the specified sandwich ingredient.",
        required_arguments=("object",),
        supported_agents=supported,
        resource_arguments=("object",),
    ))
    registry.register(SkillSpec(
        name=SKILL_STACK_ON,
        description=(
            "Place the currently-held ingredient precisely on top of the specified "
            "target (another ingredient or the cutting board). Requires the ingredient "
            "to be in-hand and the target to have a flat top surface."
        ),
        required_arguments=("target",),
        supported_agents=supported,
        resource_arguments=("target",),
    ))
    registry.register(SkillSpec(
        name=SKILL_WAIT,
        description="Hold position while the other agent acts.",
        required_arguments=(),
        supported_agents=supported,
    ))
    return registry


def build_sweep_registry(agent_names: Sequence[str]) -> SkillRegistry:
    """Skills for the sweeping task."""
    supported = tuple(agent_names)
    registry = SkillRegistry()
    registry.register(SkillSpec(
        name=SKILL_SWEEP,
        description=(
            "Move the broom end-effector in an arc to sweep the debris into the dustpan. "
            "Robot must approach the debris region before executing."
        ),
        required_arguments=("target",),
        supported_agents=supported,
        resource_arguments=("target",),
    ))
    registry.register(SkillSpec(
        name=SKILL_WAIT,
        description="Hold position while the other agent sweeps.",
        required_arguments=(),
        supported_agents=supported,
    ))
    return registry


# Mapping from task class name → registry factory
_TASK_REGISTRY_MAP = {
    "PackGroceryTask": build_pick_place_registry,
    "SortTask": build_pick_place_registry,
    "RopeTask": build_pick_place_registry,
    "OpenCabinetTask": build_cabinet_registry,
    "MakeSandwichTask": build_sandwich_registry,
    "SweepTask": build_sweep_registry,
}


def build_registry_for_task(env, agent_names: Sequence[str]) -> SkillRegistry:
    """Auto-select the right registry based on the env class name."""
    class_name = type(env).__name__
    factory = _TASK_REGISTRY_MAP.get(class_name, build_pick_place_registry)
    return factory(agent_names)
