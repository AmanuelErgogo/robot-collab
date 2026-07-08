"""RoCo backend tests.

The structural test is light (no MuJoCo). The end-to-end test drives the real
RoCoBench simulator and is opt-in: it only runs when ``ROCO_SIM_TEST=1`` and
MuJoCo is importable, and it sets ``MUJOCO_GL=egl`` for offscreen rendering.
"""

import os

import pytest


def test_roco_backend_imports_without_mujoco():
    # Importing the module must not require MuJoCo (heavy imports are lazy).
    import sys

    from rocobench.crie_bt.pipeline import roco_backend

    assert hasattr(roco_backend, "RoCoBenchEnvironmentAdapter")
    assert hasattr(roco_backend, "RoCoRRTSkillExecutorAdapter")
    assert hasattr(roco_backend, "RoCoScriptedPlanner")
    assert hasattr(roco_backend, "build_roco_step1")
    assert "mujoco" not in sys.modules


def test_roco_scripted_planner_emits_execute_block():
    from rocobench.crie_bt.pipeline.interfaces import ObservationBundle
    from rocobench.crie_bt.pipeline.roco_backend import RoCoScriptedPlanner

    block = "EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT"
    planner = RoCoScriptedPlanner([block])
    plan = planner.propose_next(ObservationBundle(), "goal", [], [])
    assert len(plan.steps) == 1
    assert plan.steps[0].agent_id == "Alice"
    assert plan.steps[0].skill_call.args["response"] == block
    # Exhausted -> empty plan (episode ends).
    assert planner.propose_next(ObservationBundle(), "goal", [], []).steps == []


@pytest.mark.skipif(
    os.environ.get("ROCO_SIM_TEST") != "1",
    reason="Set ROCO_SIM_TEST=1 to run the real RoCoBench simulator end-to-end test.",
)
def test_real_roco_step1_end_to_end():
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rocobench.crie_bt.pipeline import build_collaboration_controller, build_condition
    from rocobench.crie_bt.pipeline.roco_backend import build_roco_step1

    condition = build_condition("CRIE-BT-RR-Cent", "step1")
    env_adapter, executor, planner_factory = build_roco_step1("pack", seed=0)  # WAIT action
    controller = build_collaboration_controller(
        condition, planner_factory=planner_factory, executor=executor)
    row = controller.run_episode("CRIE-BT-RR-Cent", env_adapter, "pack", "pack", max_steps=4)

    assert row["environment"] == "sim"
    assert row["monitor_backend"] == "CodedSim"
    assert row["monitor_updates"] >= 1  # the coded monitor read real oracle state
    assert row["num_steps"] >= 1        # a real RoCo action executed through RRT
