import os

import numpy as np

from integrations.lerobot_roco.dataset.manifest import atomic_write_json
from integrations.lerobot_roco.dataset.schema import FeatureSpec, default_schema_for_tests
from integrations.lerobot_roco.training.checkpoint import (
    inspect_checkpoint,
    postprocess_action_to_native,
)
from integrations.lerobot_roco.training.evaluate_offline import compute_offline_action_metrics


def test_offline_metrics_include_chunk_horizon_and_bounds():
    predictions = np.asarray([[[0.0, 0.5], [0.2, 0.7]]], dtype=np.float32)
    targets = np.asarray([[[0.1, 0.5], [0.1, 0.6]]], dtype=np.float32)

    metrics = compute_offline_action_metrics(
        predictions,
        targets,
        action_low=[-1.0, 0.0],
        action_high=[1.0, 1.0],
        gripper_index=1,
        gripper_threshold=0.5,
    )

    assert metrics.sample_count == 1
    assert len(metrics.chunk_mae_by_horizon) == 2
    assert metrics.native_bound_violations == {"below_low": 0, "above_high": 0}
    assert metrics.gripper_accuracy == 1.0


def test_native_action_postprocess_mean_std_and_bounds():
    feature = FeatureSpec("action", (2,), "float32", low=(-2.0, -2.0), high=(2.0, 2.0))

    native = postprocess_action_to_native(
        np.asarray([0.0, 1.0], dtype=np.float32),
        feature,
        normalization_mode="mean_std",
        stats={"mean": [0.5, 0.5], "std": [0.25, 0.5]},
    )

    assert native.tolist() == [0.5, 1.0]


def test_checkpoint_inspection_reads_policy_metadata(tmp_path):
    checkpoint_dir = tmp_path / "run"
    model_dir = checkpoint_dir / "checkpoints" / "last" / "pretrained_model"
    os.makedirs(model_dir, exist_ok=True)
    schema = default_schema_for_tests()
    dataset_root = tmp_path / "dataset"
    os.makedirs(dataset_root, exist_ok=True)
    atomic_write_json(str(dataset_root / "schema.json"), schema.to_dict())
    atomic_write_json(
        str(model_dir / "config.json"),
        {
            "type": "act",
            "input_features": {
                "observation.images.active_agent": {"shape": [3, 4, 5]},
                "observation.images.front": {"shape": [3, 4, 5]},
                "observation.state": {"shape": [3]},
            },
            "output_features": {"action": {"shape": [2]}},
        },
    )
    atomic_write_json(str(model_dir / "train_config.json"), {"steps": 50})
    with open(model_dir / "model.safetensors", "wb") as f:
        f.write(b"placeholder")

    inspection = inspect_checkpoint(str(checkpoint_dir), dataset_root=str(dataset_root))

    assert inspection.ok
    assert inspection.files_present["config.json"] is True
    assert inspection.metadata["policy_config"]["type"] == "act"
