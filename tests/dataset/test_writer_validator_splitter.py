import numpy as np

import integrations.lerobot_roco.dataset.writer as writer_module
from integrations.lerobot_roco.dataset.episode_sampler import VariationSpec
from integrations.lerobot_roco.dataset.recorder import EpisodeRecord, TransitionFrame
from integrations.lerobot_roco.dataset.schema import default_schema_for_tests
from integrations.lerobot_roco.dataset.splitter import create_variation_group_splits, detect_split_leakage
from integrations.lerobot_roco.dataset.validator import validate_dataset
from integrations.lerobot_roco.dataset.writer import (
    AtomicEpisodeWriter,
    export_local_dataset_to_lerobot,
    load_native_action_payloads,
)


def make_record(schema, episode_index=0, action_value=0.1, variation=None):
    variation = variation or VariationSpec(
        seed=episode_index,
        variation_index=episode_index,
        object_name="apple",
        target_name="bin_front_left",
    )
    observation = {
        "pixels": {
            "front": np.full((4, 5, 3), 10 + episode_index, dtype=np.uint8),
            "active_agent": np.full((4, 5, 3), 20 + episode_index, dtype=np.uint8),
        },
        "agent_pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }
    frame = TransitionFrame(
        observation=observation,
        action=np.asarray([action_value, action_value], dtype=np.float32),
        timestamp=float(episode_index) * 0.1,
        frame_index=0,
        episode_index=episode_index,
        env_step_index=0,
        native_action={
            "ctrl_idxs": np.asarray([0, 1], dtype=np.int32),
            "ctrl_vals": np.asarray([action_value, action_value], dtype=np.float32),
            "qpos_idxs": np.asarray([0], dtype=np.int32),
            "qpos_target": np.asarray([action_value], dtype=np.float32),
            "eq_active_idxs": np.asarray([], dtype=np.int32),
            "eq_active_vals": np.asarray([], dtype=np.int32),
        },
    )
    metadata = {
        "episode_id": "episode_{:06d}".format(episode_index),
        "seed": variation.seed,
        "variation_id": variation.variation_id,
        "task_id": schema.task_id,
        "skill_id": schema.skill_id,
        "canonical_skill_call": "PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)",
        "natural_language_instruction": "Alice put the apple into bin_front_left.",
        "agent_name": "Alice",
        "robot_name": "fake_robot",
        "object_name": "apple",
        "target_name": "bin_front_left",
        "success": True,
        "termination_reason": "success",
        "frame_count": 1,
        "expert_backend": "rrt",
        "expert_plan_id": "plan",
        "schema_hash": schema.schema_hash,
    }
    return EpisodeRecord(
        episode_id=metadata["episode_id"],
        episode_index=episode_index,
        variation=variation,
        schema_hash=schema.schema_hash,
        frames=[frame],
        metadata=metadata,
    )


def test_atomic_writer_resume_and_validator(tmp_path):
    schema = default_schema_for_tests()
    writer = AtomicEpisodeWriter(str(tmp_path), schema)
    record = make_record(schema)

    status, episode_id = writer.write_episode(record)
    skipped_status, skipped_episode_id = writer.write_episode(record)
    create_variation_group_splits(str(tmp_path), seed=0)
    report = validate_dataset(str(tmp_path))

    assert status == "committed"
    assert episode_id == "episode_000000"
    assert skipped_status == "skipped"
    assert skipped_episode_id == "episode_000000"
    assert report.ok
    assert load_native_action_payloads(str(tmp_path / "episodes" / "episode_000000"))[0]["ctrl_idxs"].tolist() == [0, 1]


def test_writer_quarantines_failed_attempt(tmp_path):
    schema = default_schema_for_tests()
    writer = AtomicEpisodeWriter(str(tmp_path), schema)

    path = writer.quarantine_episode("episode_bad", "POSTCONDITION_FAILED", "object not packed")

    assert path.endswith("episode_bad")
    assert writer.episode_ids() == []


def test_splitter_groups_by_variation_without_leakage(tmp_path):
    schema = default_schema_for_tests()
    writer = AtomicEpisodeWriter(str(tmp_path), schema)
    for index in range(4):
        writer.write_episode(make_record(schema, episode_index=index))

    manifest = create_variation_group_splits(str(tmp_path), seed=123)

    assert not detect_split_leakage(manifest.to_dict())
    leaked = {
        "variation_ids": {
            "train": ["var_a"],
            "validation": ["var_a"],
        }
    }
    assert detect_split_leakage(leaked) == ["var_a"]


def test_validator_detects_action_bounds(tmp_path):
    schema = default_schema_for_tests()
    writer = AtomicEpisodeWriter(str(tmp_path), schema)
    record = make_record(schema, action_value=2.0)
    writer.write_episode(record)
    create_variation_group_splits(str(tmp_path), seed=0)

    report = validate_dataset(str(tmp_path))

    assert not report.ok
    assert any(issue.code == "ACTION_OUT_OF_BOUNDS" for issue in report.issues)


def test_export_local_dataset_uses_public_lerobot_flow(tmp_path, monkeypatch):
    schema = default_schema_for_tests()
    local_root = tmp_path / "local"
    writer = AtomicEpisodeWriter(str(local_root), schema)
    writer.write_episode(make_record(schema))
    calls = []

    class FakeDataset:
        @classmethod
        def create(cls, **kwargs):
            calls.append(("create", kwargs))
            return cls()

        def add_frame(self, frame, task=None):
            calls.append(("add_frame", sorted(frame.keys()), task))

        def save_episode(self):
            calls.append(("save_episode",))

        def finalize(self):
            calls.append(("finalize",))

    monkeypatch.setattr(writer_module, "_import_lerobot_dataset", lambda: FakeDataset)

    export_local_dataset_to_lerobot(
        str(local_root),
        str(tmp_path / "lerobot"),
        repo_id="local/test",
        robot_type="fake_robot",
        use_videos=False,
    )

    assert calls[0][0] == "create"
    assert any(call[0] == "add_frame" for call in calls)
    assert calls[-1] == ("finalize",)
