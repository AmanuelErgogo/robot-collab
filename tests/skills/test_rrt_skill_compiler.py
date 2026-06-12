import pytest

from rocobench.skills.compiler import RRTSkillCompiler, SkillCompilationError
from rocobench.skills.models import SkillCall, SkillPlan
from fakes import FakeObs, FakePackEnv


class SpyParser:
    def __init__(self, success=True):
        self.success = success
        self.responses = []

    def parse(self, obs, response):
        self.responses.append(response)
        if not self.success:
            return False, "legacy failed", []
        return True, "parsed", ["compiled-plan"]


def put(agent, obj, slot):
    return SkillCall(agent, "PUT_OBJECT_IN_CONTAINER", {"object": obj, "container": slot}, "raw")


def wait(agent):
    return SkillCall(agent, "WAIT", {}, "raw")


def test_compiler_builds_exact_pick_place_synthetic_response():
    env = FakePackEnv()
    parser = SpyParser()
    compiler = RRTSkillCompiler(env, parser)
    plan = SkillPlan([put("Alice", "apple", "bin_front_left"), wait("Bob")], "EXECUTE")

    prepared = compiler.compile(plan, FakeObs())

    assert parser.responses[0] == (
        "EXECUTE\n"
        "NAME Alice ACTION PICK apple PLACE bin_front_left\n"
        "NAME Bob ACTION WAIT"
    )
    assert prepared.backend_name == "rrt"
    assert prepared.compiled_plans == ["compiled-plan"]
    assert prepared.metadata["synthetic_response"] == parser.responses[0]


def test_compiler_builds_place_only_when_agent_already_holds_object():
    env = FakePackEnv()
    parser = SpyParser()
    compiler = RRTSkillCompiler(env, parser)
    plan = SkillPlan([put("Alice", "apple", "bin_front_left"), wait("Bob")], "EXECUTE")

    compiler.compile(plan, FakeObs(alice_contacts={"apple"}))

    assert "NAME Alice ACTION PLACE apple bin_front_left" in parser.responses[0]


def test_compiler_propagates_legacy_parser_failure():
    env = FakePackEnv()
    parser = SpyParser(success=False)
    compiler = RRTSkillCompiler(env, parser)
    plan = SkillPlan([put("Alice", "apple", "bin_front_left"), wait("Bob")], "EXECUTE")

    with pytest.raises(SkillCompilationError) as exc_info:
        compiler.compile(plan, FakeObs())

    assert exc_info.value.code == "COMPILATION_FAILED"
    assert "legacy failed" in exc_info.value.message
