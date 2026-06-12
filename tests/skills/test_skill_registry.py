import pytest

from rocobench.skills.models import SkillCall, SkillSpec
from rocobench.skills.pack_grocery import build_pack_grocery_skill_registry
from rocobench.skills.registry import SkillRegistry


def test_registry_aliases_and_agent_prompt_are_deterministic():
    registry = SkillRegistry()
    registry.register(SkillSpec(
        name="WAIT",
        description="hold",
        required_arguments=(),
        aliases=("idle",),
    ))

    assert registry.has("idle")
    assert registry.get("idle").name == "WAIT"
    assert registry.render_agent_skill_prompt("Alice").startswith("1. WAIT()")


def test_registry_rejects_alias_collision():
    registry = SkillRegistry()
    registry.register(SkillSpec("WAIT", "hold", ()))

    with pytest.raises(ValueError):
        registry.register(SkillSpec("MOVE", "move", (), aliases=("wait",)))


def test_registry_validates_shape():
    registry = build_pack_grocery_skill_registry(["Alice", "Bob"])
    missing = SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple"}, "raw")
    unknown = SkillCall("Alice", "WAIT", {"object": "apple"}, "raw")
    unsupported = SkillCall("Chad", "WAIT", {}, "raw")

    assert registry.validate_call_shape(missing).issues[0].code == "MISSING_ARGUMENT"
    assert registry.validate_call_shape(unknown).issues[0].code == "UNKNOWN_ARGUMENT"
    assert registry.validate_call_shape(unsupported).issues[0].code == "UNSUPPORTED_AGENT"
