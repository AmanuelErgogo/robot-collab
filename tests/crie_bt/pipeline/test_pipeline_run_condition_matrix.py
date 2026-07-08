"""Smoke tests for the condition-matrix runner and analysis scripts."""

import importlib.util
import json
import os

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))


def _load(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_SCRIPTS, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_emits_full_condition_metadata(tmp_path):
    runner = _load("run_condition_matrix", "run_condition_matrix.py")
    out = os.path.join(str(tmp_path), "dry.jsonl")
    runner.main(["--stage", "step1", "--dry-run", "--episodes", "1", "--output", out])
    with open(out, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    # step1 defaults to 4 robot-robot conditions x 4 tasks x 1 episode.
    assert len(rows) == 16
    assert {row["condition_name"] for row in rows} == {
        "VLM-RR-Cent", "VLM-RR-Dialog", "CRIE-BT-RR-Cent", "CRIE-BT-RR-Dialog"}
    # Every dry row carries the full condition metadata.
    for row in rows:
        for field in ("condition_name", "controller_family", "team_type", "coordination_mode",
                      "skill_backend", "monitor_backend", "monitor_privileged", "environment",
                      "planner_input_type"):
            assert field in row


def test_synthetic_run_and_analysis(tmp_path):
    runner = _load("run_condition_matrix", "run_condition_matrix.py")
    out = os.path.join(str(tmp_path), "results.jsonl")
    runner.main([
        "--stage", "step1",
        "--conditions", "VLM-RR-Cent", "CRIE-BT-RR-Cent",
        "--tasks", "sandwich",
        "--episodes", "2", "--output", out,
    ])
    with open(out, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 4
    assert all(row["success"] for row in rows)

    analyzer = _load("analyze_condition_matrix", "analyze_condition_matrix.py")
    analysis_dir = os.path.join(str(tmp_path), "analysis")
    analyzer.main([out, "--output-dir", analysis_dir])
    assert os.path.exists(os.path.join(analysis_dir, "summary.csv"))
    assert os.path.exists(os.path.join(analysis_dir, "summary.tex"))
