from rocobench.skills.models import SkillCall, SkillPlan
from rocobench.skills.pack_grocery import build_pack_grocery_skill_registry
from rocobench.skills.validation import PackGrocerySkillPlanValidator
from fakes import FakeObs, FakePackEnv


def plan(*calls):
    return SkillPlan(calls=list(calls), parsed_proposal="EXECUTE")


def put(agent, obj, slot):
    return SkillCall(agent, "PUT_OBJECT_IN_CONTAINER", {"object": obj, "container": slot}, "raw")


def wait(agent):
    return SkillCall(agent, "WAIT", {}, "raw")


def make_validator(env):
    return PackGrocerySkillPlanValidator(
        env,
        build_pack_grocery_skill_registry(["Alice", "Bob"]),
        ["Alice", "Bob"],
    )


def issue_codes(result):
    return [issue.code for issue in result.issues]


def test_valid_put_wait_and_two_agent_plan():
    env = FakePackEnv()
    obs = FakeObs()
    validator = make_validator(env)

    assert validator.validate(plan(put("Alice", "apple", "bin_front_left"), wait("Bob")), obs).valid
    assert validator.validate(
        plan(
            put("Alice", "apple", "bin_front_left"),
            put("Bob", "banana", "bin_front_right"),
        ),
        obs,
    ).valid


def test_resource_conflicts_and_unknowns():
    env = FakePackEnv()
    obs = FakeObs()
    validator = make_validator(env)

    assert "RESOURCE_CONFLICT" in issue_codes(
        validator.validate(plan(put("Alice", "apple", "bin_front_left"), put("Bob", "apple", "bin_front_right")), obs)
    )
    assert "RESOURCE_CONFLICT" in issue_codes(
        validator.validate(plan(put("Alice", "apple", "bin_front_left"), put("Bob", "banana", "bin_front_left")), obs)
    )
    assert "UNKNOWN_OBJECT" in issue_codes(
        validator.validate(plan(put("Alice", "orange", "bin_front_left"), wait("Bob")), obs)
    )
    assert "UNKNOWN_TARGET" in issue_codes(
        validator.validate(plan(put("Alice", "apple", "bin_missing"), wait("Bob")), obs)
    )


def test_packed_occupied_and_holding_recovery():
    env = FakePackEnv()
    obs = FakeObs(alice_contacts={"apple"}, bob_contacts={"banana"})
    validator = make_validator(env)

    assert validator.validate(plan(put("Alice", "apple", "bin_front_left"), wait("Bob")), obs).valid
    assert "AGENT_HOLDING_DIFFERENT_OBJECT" in issue_codes(
        validator.validate(plan(put("Bob", "apple", "bin_front_left"), wait("Alice")), obs)
    )
    assert "OBJECT_HELD_BY_OTHER_AGENT" in issue_codes(
        validator.validate(plan(put("Bob", "apple", "bin_front_left"), wait("Alice")), obs)
    )

    env.occupancy["bin_front_left"] = "apple"
    result = validator.validate(plan(put("Alice", "apple", "bin_front_left"), wait("Bob")), obs)
    codes = issue_codes(result)
    assert "OBJECT_ALREADY_PACKED" in codes
    assert "TARGET_OCCUPIED" in codes


def test_all_wait_rejected():
    env = FakePackEnv()
    result = make_validator(env).validate(plan(wait("Alice"), wait("Bob")), FakeObs())

    assert not result.valid
    assert "NO_PROGRESS" in issue_codes(result)
