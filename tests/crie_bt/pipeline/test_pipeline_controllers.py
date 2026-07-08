"""Controller behaviour and episode-logging tests (plan tests 2, 7)."""

import json
import os

from rocobench.crie_bt.pipeline import (
    ScriptedHumanCommunicationInterface,
    SyntheticEnvironmentAdapter,
    build_collaboration_controller,
    build_condition,
)
from rocobench.crie_bt.pipeline.controllers import (
    CRIEBTCentralizedController,
    VLMCentralizedController,
)
from rocobench.crie_bt.pipeline.episode_logger import REQUIRED_ROW_FIELDS

_PLAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "crie_next_stage_plan"))


def _run(condition_name, stage, task, fail=None, comm=None, max_steps=80, seed=0, episode_index=0):
    cfg = build_condition(condition_name, stage)
    env = SyntheticEnvironmentAdapter(fail_stage_ids=fail)
    controller = build_collaboration_controller(cfg, communication=comm)
    return controller, controller.run_episode(
        condition_name, env, task, task, max_steps, seed=seed, episode_index=episode_index)


def test_baseline_has_no_explicit_monitor():
    controller, row = _run("VLM-RR-Cent", "step1", "sandwich")
    assert isinstance(controller, VLMCentralizedController)
    assert controller.uses_explicit_monitor is False
    assert controller.monitor is None
    assert row["monitor_updates"] == 0
    assert row["monitor_backend"] == "VLM-self"


def test_criebt_runs_explicit_monitor():
    controller, row = _run("CRIE-BT-RR-Cent", "step1", "sandwich")
    assert isinstance(controller, CRIEBTCentralizedController)
    assert controller.uses_explicit_monitor is True
    assert controller.monitor is not None
    assert row["monitor_updates"] > 0
    assert row["monitor_backend"] == "CodedSim"
    assert row["monitor_privileged"] is True


def test_all_four_step1_conditions_complete_clean_task():
    for name in ("VLM-RR-Cent", "VLM-RR-Dialog", "CRIE-BT-RR-Cent", "CRIE-BT-RR-Dialog"):
        _, row = _run(name, "step1", "sandwich")
        assert row["success"] is True
        assert row["task_done"] is True


def test_criebt_recovers_with_local_retries_baseline_replans():
    # The same injected failure is recovered by CRIE-BT via cheap local retries
    # and by the baseline via (more expensive) planner replans.
    fail = {"place_cheese": 2}
    _, vlm = _run("VLM-RR-Cent", "step1", "sandwich", fail=fail)
    _, crie = _run("CRIE-BT-RR-Cent", "step1", "sandwich", fail=fail)
    assert vlm["success"] and crie["success"]
    assert vlm["replans"] >= 2 and vlm["local_retries"] == 0
    assert crie["local_retries"] >= 1
    assert crie["planner_calls"] < vlm["planner_calls"]


def test_human_robot_episode_completes_with_scripted_human():
    comm = ScriptedHumanCommunicationInterface(default_action="accept")
    _, row = _run("CRIE-BT-HR-Cent", "step2", "medication_sim", comm=comm)
    assert row["success"] is True
    assert row["team_type"] == "HR"


def test_dialog_condition_logs_dialogue_turns():
    comm = ScriptedHumanCommunicationInterface(default_action="accept")
    _, row = _run("VLM-HR-Dialog", "step2", "medication_sim", comm=comm)
    assert row["dialogue_turns"] > 0


def test_human_counter_propose_counts_as_intervention():
    comm = ScriptedHumanCommunicationInterface(responses=["counter do it differently", "accept"],
                                               default_action="accept")
    _, row = _run("CRIE-BT-HR-Cent", "step2", "medication_sim", comm=comm)
    assert row["human_interventions"] >= 1
    assert row["success"] is True


def test_episode_log_contains_required_metadata():
    _, row = _run("CRIE-BT-RR-Cent", "step1", "pack", seed=3, episode_index=5)
    for field in REQUIRED_ROW_FIELDS:
        assert field in row, "missing required field {!r}".format(field)
    # Cross-check against the machine-readable logging schema from the plan.
    with open(os.path.join(_PLAN_DIR, "logging_schema.json"), "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    for field in schema["required"]:
        assert field in row, "row missing schema-required field {!r}".format(field)
    assert row["episode_id"] == "step1_pack_seed3_ep005_criebt_rr_cent"
    assert row["condition_name"] == "CRIE-BT-RR-Cent"


def test_events_file_is_written(tmp_path):
    cfg = build_condition("CRIE-BT-RR-Cent", "step1")
    env = SyntheticEnvironmentAdapter(fail_stage_ids={"place_cheese": 1})
    controller = build_collaboration_controller(cfg)
    events_path = os.path.join(str(tmp_path), "events.jsonl")
    controller.run_episode("CRIE-BT-RR-Cent", env, "sandwich", "sandwich", 80, events_path=events_path)
    assert os.path.exists(events_path)
    with open(events_path, "r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    types = {event["event_type"] for event in events}
    assert {"episode_start", "planner_call_start", "skill_start", "monitor_update", "episode_end"} <= types
    assert any(event["event_type"] == "local_retry" for event in events)
