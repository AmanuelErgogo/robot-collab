from dataclasses import dataclass

import numpy as np

from integrations.lerobot_roco.evaluation.metrics import aggregate_rollout_results, wilson_interval
from integrations.lerobot_roco.evaluation.policy_loader import validate_native_action_chunk


@dataclass
class DummyResult:
    success: bool
    termination_reason: str
    num_env_steps: int
    inference_latency_ms: tuple
    action_bound_violations: int = 0
    no_progress_events: int = 0
    final_info: dict = None


def test_native_action_validation_counts_bounds_and_shape():
    valid = validate_native_action_chunk(
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        action_low=[-1.0, 0.0],
        action_high=[1.0, 2.0],
        expected_action_dim=2,
    )
    assert valid.ok

    invalid = validate_native_action_chunk(
        np.asarray([[2.0, -1.0]], dtype=np.float32),
        action_low=[-1.0, 0.0],
        action_high=[1.0, 2.0],
        expected_action_dim=2,
    )
    assert invalid.violation_count == 2

    wrong_shape = validate_native_action_chunk(
        np.zeros((1, 3), dtype=np.float32),
        action_low=[-1.0, 0.0],
        action_high=[1.0, 2.0],
        expected_action_dim=2,
    )
    assert wrong_shape.shape_ok is False


def test_wilson_and_aggregate_metrics():
    interval = wilson_interval(1, 2)
    assert 0.0 <= interval["low"] <= interval["high"] <= 1.0

    metrics = aggregate_rollout_results(
        [
            DummyResult(True, "SUCCESS", 2, (1.0, 3.0), final_info={}),
            DummyResult(False, "MAX_STEPS", 4, (5.0,), action_bound_violations=1, final_info={}),
        ]
    )

    assert metrics.episode_count == 2
    assert metrics.success_rate == 0.5
    assert metrics.termination_distribution == {"SUCCESS": 1, "MAX_STEPS": 1}
    assert metrics.latency_ms["p50"] == 3.0
    assert metrics.action_bound_violations == 1

