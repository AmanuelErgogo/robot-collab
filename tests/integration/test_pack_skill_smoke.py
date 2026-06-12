import pytest


pytest.importorskip("dm_control")

from prompting.skill_feedback import SkillFeedbackManager
from prompting.skill_parser import SkillResponseParser
from rocobench.envs import PackGroceryTask
from rocobench.skills import RRTSkillCompiler, build_pack_grocery_skill_registry
from rocobench.skills.validation import PackGrocerySkillPlanValidator
from prompting.feedback import FeedbackManager
from prompting.parser import LLMResponseParser
from rocobench.rrt_multi_arm import MultiArmRRT


@pytest.mark.integration
def test_pack_skill_parse_validate_compile_and_geometric_feedback():
    env = PackGroceryTask(
        render_cameras=["teaser"],
        randomize_init=False,
        render_point_cloud=False,
    )
    obs = env.get_obs()
    agent_names = list(env.robot_name_map.values())
    registry = build_pack_grocery_skill_registry(agent_names)
    parser = SkillResponseParser(registry, agent_names)
    success, _, plans = parser.parse(
        obs,
        "EXECUTE\n"
        "NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)\n"
        "NAME Bob ACTION WAIT()",
    )
    assert success

    legacy_parser = LLMResponseParser(env, "action_only", env.robot_name_map, ["NAME", "ACTION"])
    planner = MultiArmRRT(
        env.physics,
        robots=env.get_sim_robots(),
        graspable_object_names=env.get_graspable_objects(),
        allowed_collision_pairs=env.get_allowed_collision_pairs(),
    )
    manager = SkillFeedbackManager(
        PackGrocerySkillPlanValidator(env, registry, agent_names),
        RRTSkillCompiler(env, legacy_parser),
        FeedbackManager(env, planner, "action_only", env.robot_name_map),
    )
    manager.update_obs(obs)
    ready, feedback = manager.give_feedback(plans[0])

    assert ready, feedback
    assert plans[0].prepared_execution is not None
    assert len(plans[0].prepared_execution.compiled_plans) > 0
