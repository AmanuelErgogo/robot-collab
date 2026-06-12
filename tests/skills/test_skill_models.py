from rocobench.skills.models import PreparedSkillExecution, SkillCall, SkillPlan, SkillSpec


def test_skill_spec_normalizes_and_rejects_duplicate_arguments():
    spec = SkillSpec(
        name="put object in container",
        description="desc",
        required_arguments=("object", "container"),
        aliases=("put",),
    )

    assert spec.name == "PUT_OBJECT_IN_CONTAINER"
    assert spec.aliases == ("PUT",)


def test_skill_plan_id_and_serialization_excludes_preparation():
    call = SkillCall(
        agent_name="Alice",
        skill_name="put_object_in_container",
        arguments={"object": "apple", "container": "bin_front_left"},
        raw_action="raw",
    )
    plan1 = SkillPlan(calls=[call], parsed_proposal="EXECUTE")
    plan2 = SkillPlan(calls=[call], parsed_proposal="different text")
    plan1.prepared_execution = PreparedSkillExecution("rrt", plan1.plan_id, ["backend-plan"])

    assert str(call) == "PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)"
    assert plan1.plan_id == plan2.plan_id
    assert "prepared_execution" not in plan1.to_dict()
    assert "NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)" in plan1.get_action_desp()
