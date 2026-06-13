import os

from integrations.lerobot_roco.dataset.manifest import DatasetManifest, atomic_write_json, read_json
from integrations.lerobot_roco.dataset.schema import default_schema_for_tests
from integrations.lerobot_roco.training.config import debug_training_config
from integrations.lerobot_roco.training.launch import launch_training


def test_dry_run_writes_manifest_without_lerobot(tmp_path):
    dataset_root = tmp_path / "dataset"
    schema = default_schema_for_tests()
    os.makedirs(dataset_root, exist_ok=True)
    atomic_write_json(str(dataset_root / "schema.json"), schema.to_dict())
    atomic_write_json(str(dataset_root / "dataset_manifest.json"), DatasetManifest.create(schema).to_dict())

    config = debug_training_config().with_overrides(
        dataset_root=str(dataset_root),
        lerobot_dataset_root=str(tmp_path / "missing_lerobot"),
        output_root=str(tmp_path / "runs"),
        dry_run=True,
        require_lerobot_dataset=False,
        require_lerobot_package=False,
        compatibility_lock=None,
    )

    result = launch_training(config)
    manifest = read_json(result.manifest_path)

    assert result.status == "dry_run"
    assert manifest["status"] == "dry_run"
    assert manifest["preflight"]["ok"] is True
    assert manifest["command"][0] == "lerobot-train"
