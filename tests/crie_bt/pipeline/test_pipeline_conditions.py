"""Condition-registry and stage/backend-mapping tests (plan tests 1, 3, 4)."""

from rocobench.crie_bt.pipeline import build_condition, get_registry
from rocobench.crie_bt.pipeline.controllers import build_monitor_for_condition
from rocobench.crie_bt.pipeline.monitors import CodedSimProgressMonitor, SARMProgressMonitor

EIGHT_CONDITIONS = [
    "VLM-RR-Cent", "VLM-RR-Dialog", "VLM-HR-Cent", "VLM-HR-Dialog",
    "CRIE-BT-RR-Cent", "CRIE-BT-RR-Dialog", "CRIE-BT-HR-Cent", "CRIE-BT-HR-Dialog",
]


def test_condition_registry_loads_all_8_conditions():
    names = get_registry().condition_names()
    assert names == EIGHT_CONDITIONS


def test_conditions_resolve_by_code_name():
    cfg = build_condition("criebt_rr_cent", "step1")
    assert cfg.condition_name == "CRIE-BT-RR-Cent"
    assert cfg.code_name == "criebt_rr_cent"


def test_criebt_uses_coded_monitor_in_step1_and_step2():
    for stage in ("step1", "step2", "step2a"):
        for name in ("CRIE-BT-RR-Cent", "CRIE-BT-RR-Dialog", "CRIE-BT-HR-Cent", "CRIE-BT-HR-Dialog"):
            cfg = build_condition(name, stage)
            if cfg.team_type == "RR" and stage.startswith("step2"):
                continue  # step2 is human-robot; RR conditions are not run there
            assert cfg.monitor_backend == "CodedSim"
            assert cfg.monitor_privileged is True
            monitor = build_monitor_for_condition(cfg)
            assert isinstance(monitor, CodedSimProgressMonitor)


def test_criebt_uses_sarm_monitor_in_step3():
    for name in ("CRIE-BT-HR-Cent", "CRIE-BT-HR-Dialog"):
        cfg = build_condition(name, "step3")
        assert cfg.monitor_backend == "SARM"
        assert cfg.monitor_privileged is False
        assert cfg.skill_backend == "LearnedSkill"
        assert cfg.environment == "real"
        monitor = build_monitor_for_condition(cfg)
        assert isinstance(monitor, SARMProgressMonitor)


def test_baseline_monitor_is_vlm_self_across_stages():
    for stage in ("step1", "step2", "step3"):
        for name in ("VLM-RR-Cent", "VLM-HR-Dialog"):
            cfg = build_condition(name, stage)
            assert cfg.monitor_backend == "VLM-self"
            assert cfg.monitor_privileged is False
