import os

from integrations.lerobot_roco.dataset.manifest import DatasetManifest, atomic_write_json
from integrations.lerobot_roco.dataset.schema import default_schema_for_tests
from integrations.lerobot_roco.training.feature_contract import run_feature_preflight


def _write_dataset(root, schema=None):
    schema = schema or default_schema_for_tests()
    os.makedirs(root, exist_ok=True)
    atomic_write_json(os.path.join(root, "schema.json"), schema.to_dict())
    atomic_write_json(os.path.join(root, "dataset_manifest.json"), DatasetManifest.create(schema).to_dict())
    return schema


def test_preflight_passes_without_required_lerobot_export(tmp_path):
    dataset_root = tmp_path / "dataset"
    schema = _write_dataset(str(dataset_root))

    result = run_feature_preflight(
        str(dataset_root),
        lerobot_dataset_root=str(tmp_path / "missing_lerobot"),
        require_lerobot_dataset=False,
    )

    assert result.ok
    assert result.contract.dataset_schema_hash == schema.schema_hash
    assert any(issue.code == "LEROBOT_DATASET_MISSING" and issue.severity == "warning" for issue in result.issues)


def test_preflight_detects_lerobot_feature_shape_mismatch(tmp_path):
    dataset_root = tmp_path / "dataset"
    lerobot_root = tmp_path / "lerobot"
    schema = _write_dataset(str(dataset_root))
    os.makedirs(lerobot_root / "meta", exist_ok=True)
    features = schema.to_lerobot_features()
    features["action"] = dict(features["action"])
    features["action"]["shape"] = [99]
    atomic_write_json(
        str(lerobot_root / "meta" / "info.json"),
        {"fps": schema.fps, "features": features},
    )
    atomic_write_json(str(lerobot_root / "meta" / "stats.json"), {})

    result = run_feature_preflight(str(dataset_root), lerobot_dataset_root=str(lerobot_root))

    assert not result.ok
    assert any(issue.code == "LEROBOT_FEATURE_SHAPE_MISMATCH" for issue in result.issues)
