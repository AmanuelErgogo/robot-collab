from typing import Sequence

from .models import SkillSpec
from .registry import SkillRegistry


PUT_OBJECT_IN_CONTAINER = "PUT_OBJECT_IN_CONTAINER"
WAIT = "WAIT"


def build_pack_grocery_skill_registry(agent_names: Sequence[str]) -> SkillRegistry:
    """Build the Phase 1 grocery skill registry for the configured agents."""
    supported = tuple(agent_names)
    registry = SkillRegistry()
    registry.register(SkillSpec(
        name=PUT_OBJECT_IN_CONTAINER,
        description=(
            "Pick the specified grocery object if necessary, transport it, "
            "place it into the specified bin slot, and release it."
        ),
        required_arguments=("object", "container"),
        supported_agents=supported,
        resource_arguments=("object", "container"),
    ))
    registry.register(SkillSpec(
        name=WAIT,
        description="Keep the robot stationary while another agent executes its skill.",
        required_arguments=(),
        supported_agents=supported,
    ))
    return registry
