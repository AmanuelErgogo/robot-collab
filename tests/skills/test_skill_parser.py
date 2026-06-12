import os

from prompting.skill_parser import SkillResponseParser
from rocobench.skills.pack_grocery import build_pack_grocery_skill_registry


def make_parser():
    return SkillResponseParser(
        registry=build_pack_grocery_skill_registry(["Alice", "Bob"]),
        agent_names=["Alice", "Bob"],
    )


def test_parser_accepts_valid_plan_and_agent_order():
    response = """Discussion first.
EXECUTE
NAME Bob ACTION WAIT()
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
"""
    success, parsed, plans = make_parser().parse(None, response)

    assert success
    assert len(plans) == 1
    assert [call.agent_name for call in plans[0].calls] == ["Alice", "Bob"]
    assert plans[0].calls[0].skill_name == "PUT_OBJECT_IN_CONTAINER"
    assert plans[0].calls[1].skill_name == "WAIT"


def test_parser_accepts_multiline_call():
    response = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(
    object=apple,
    container=bin_front_left
)
NAME Bob ACTION WAIT()
"""
    success, _, plans = make_parser().parse(None, response)

    assert success
    assert plans[0].calls[0].arguments["container"] == "bin_front_left"


def test_parser_failures():
    cases = [
        ("NAME Alice ACTION WAIT()", "EXECUTE"),
        ("EXECUTE\nNAME Alice ACTION WAIT()", "Missing ACTION line"),
        ("EXECUTE\nNAME Alice ACTION WAIT()\nNAME Alice ACTION WAIT()", "Duplicate"),
        ("EXECUTE\nNAME Chad ACTION WAIT()\nNAME Bob ACTION WAIT()", "Unknown agent"),
        ("EXECUTE\nNAME Alice ACTION UNKNOWN_SKILL()\nNAME Bob ACTION WAIT()", "Unknown skill"),
        ("EXECUTE\nNAME Alice ACTION PUT_OBJECT_IN_CONTAINER(apple, bin_front_left)\nNAME Bob ACTION WAIT()", "Positional"),
        ("EXECUTE\nNAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple)\nNAME Bob ACTION WAIT()", "MISSING_ARGUMENT"),
        ("EXECUTE\nNAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, object=banana, container=bin_front_left)\nNAME Bob ACTION WAIT()", "Duplicate argument"),
        ("EXECUTE\nNAME Alice ACTION WAIT(object=apple)\nNAME Bob ACTION WAIT()", "UNKNOWN_ARGUMENT"),
        ("EXECUTE\nNAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container)\nNAME Bob ACTION WAIT()", "key=value"),
        ("EXECUTE\nNAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left\nNAME Bob ACTION WAIT()", "Malformed parentheses"),
    ]
    parser = make_parser()

    for response, expected in cases:
        success, message, plans = parser.parse(None, response)
        assert not success, response
        assert expected in message
        assert plans == []


def test_parser_does_not_execute_quoted_values():
    marker = "/tmp/roco_skill_parser_marker"
    if os.path.exists(marker):
        os.remove(marker)
    response = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object="__import__('os').system('touch /tmp/roco_skill_parser_marker')", container=bin_front_left)
NAME Bob ACTION WAIT()
"""
    success, _, plans = make_parser().parse(None, response)

    assert success
    assert not os.path.exists(marker)
    assert plans[0].calls[0].arguments["object"].startswith("__import__")
