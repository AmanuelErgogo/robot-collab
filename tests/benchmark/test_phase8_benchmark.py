import csv
import json
import os

import numpy as np
import pytest

from benchmark.rocobench_pack_v1.config import EvaluationRunConfig
from benchmark.rocobench_pack_v1.evaluator import BenchmarkEvaluator
from benchmark.rocobench_pack_v1.metrics import paired_comparison, wilson_interval
from benchmark.rocobench_pack_v1.processors import BenchmarkObservationProcessor, BenchmarkProcessorError
from benchmark.rocobench_pack_v1.tasks import load_tasks
from benchmark.rocobench_pack_v1.validator import validate_release
from benchmark.rocobench_pack_v1.variations import load_variations, manifest_hash, select_variations


def test_manifests_are_versioned_and_variations_are_hashed():
    tasks = load_tasks()
    variations = load_variations()

    assert {task.task_id for task in tasks} >= {
        "pack.put.alice",
        "pack.put.bob",
        "pack.sequential.two_agent",
        "pack.concurrent.safe_two_agent",
    }
    assert len(manifest_hash()) == 64
    assert all(len(variation.variation_hash) == 64 for variation in variations)
    groups = {variation.variation_group for variation in variations}
    assert all(task.variation_group in groups for task in tasks)


def test_select_variations_reuses_identical_order_for_methods():
    first = select_variations("pack_put_alice_v1", 3)
    second = select_variations("pack_put_alice_v1", 3)

    assert [item.variation_id for item in first] == [item.variation_id for item in second]
    assert [item.seed for item in first] == [2840744289, 8101, 2840744289]


def test_processor_maps_raw_keys_without_normalizing_images():
    processor = BenchmarkObservationProcessor(required_cameras=("front", "active_agent"))
    obs = {
        "agent_pos": np.asarray([1.0, 2.0], dtype=np.float32),
        "pixels": {
            "front": np.full((4, 5, 3), 255, dtype=np.uint8),
            "active_agent": np.zeros((4, 5, 3), dtype=np.uint8),
        },
    }

    processed = processor.process(obs)

    assert processed["observation.images.front"].dtype == np.uint8
    assert int(processed["observation.images.front"].max()) == 255
    assert processed["observation.state"].dtype == np.float32
    assert processor.last_trace.to_dict()["camera_shapes"]["front"] == [4, 5, 3]


def test_processor_rejects_missing_camera():
    processor = BenchmarkObservationProcessor(required_cameras=("front", "active_agent"))
    with pytest.raises(BenchmarkProcessorError):
        processor.process({"agent_pos": np.zeros((2,), dtype=np.float32), "pixels": {"front": np.zeros((4, 5, 3), dtype=np.uint8)}})


def test_wilson_interval_and_paired_comparison_are_explicit():
    interval = wilson_interval(3, 5)
    assert 0.0 <= interval["low"] <= interval["high"] <= 1.0
    comparison = paired_comparison(
        {
            "a": {"success": "true"},
            "b": {"success": "false"},
        },
        {
            "a": {"success": "false"},
            "b": {"success": "false"},
        },
    )
    assert comparison["paired_n"] == 2
    assert comparison["mcnemar"]["b10"] == 1


def test_evaluator_writes_manifest_raw_rows_and_metrics(tmp_path, monkeypatch):
    class FakeBaseline:
        def run(self, task, variation, episode_index, episode_dir):
            os.makedirs(episode_dir, exist_ok=True)
            return {
                "success": True,
                "overall_success": True,
                "learned_success": False,
                "fallback_success": False,
                "termination_reason": "SUCCESS",
                "num_env_steps": 1,
                "sim_time_s": 0.05,
                "artifact_dir": episode_dir,
                "extra": {"episode_index": episode_index, "task": task.task_id, "variation": variation.variation_id},
            }

    monkeypatch.setattr("benchmark.rocobench_pack_v1.evaluator.build_baseline", lambda config: FakeBaseline())
    config = EvaluationRunConfig(
        method="hold",
        task_ids=("pack.put.alice",),
        episodes_per_task=1,
        output_dir=str(tmp_path),
        overwrite=True,
        require_bridge=False,
    )

    run_manifest, results, metrics = BenchmarkEvaluator(config).evaluate()

    assert run_manifest["variation_manifest_hash"] == manifest_hash()
    assert len(results) == 1
    assert metrics["success_rate"] == 1.0
    with open(tmp_path / "episodes.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["track"] == "skill_policy"
    assert rows[0]["learned_success"] == "False"
    assert rows[0]["fallback_success"] == "False"
    with open(tmp_path / "run_manifest.json", "r", encoding="utf-8") as f:
        assert json.load(f)["episodes"][0]["variation_hash"] == results[0].variation_hash


def test_release_validator_tier1_passes_with_bridge_blocked(tmp_path):
    result = validate_release(output_path=str(tmp_path / "validation.json"))

    assert result["status"] == "pass"
    assert any(check["name"] == "bridge_reset_step_render" and check["status"] == "blocked" for check in result["checks"])
    assert (tmp_path / "validation.json").exists()
