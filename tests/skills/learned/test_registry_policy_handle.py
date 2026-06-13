import os

import numpy as np
import pytest

from rocobench.skills.learned.errors import PolicyHandleError, PolicyRegistryError
from rocobench.skills.learned.models import LearnedPolicySpec
from rocobench.skills.learned.policy_handle import BoundedPolicyHandleCache, LearnedPolicyHandle, NativeActionChunk
from rocobench.skills.learned.registry import LearnedPolicyRegistry


def make_spec(tmp_path, policy_id="p0", agent="Alice", schema_hash="schema"):
    checkpoint = tmp_path / policy_id
    checkpoint.mkdir()
    return LearnedPolicySpec(
        policy_id=policy_id,
        skill_name="PUT_OBJECT_IN_CONTAINER",
        agent_name=agent,
        embodiment_id="ur5e_robotiq",
        task_id="pack",
        checkpoint=str(checkpoint),
        checkpoint_revision="rev",
        policy_type="ACT",
        schema_hash=schema_hash,
        action_representation="absolute_joint_position_plus_gripper",
        cameras=("front", "active_agent"),
        max_steps=5,
        execution_horizon=1,
        success_monitor="stable",
        failure_monitors=("NO_PROGRESS", "POLICY_INFERENCE_FAILURE"),
    )


class FakeHandle(LearnedPolicyHandle):
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.unloaded = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict_native_chunk(self, observation, instruction, action_low, action_high):
        return NativeActionChunk(np.zeros((2, 2), dtype=np.float32), {"source": "fake"})

    def health_check(self, spec):
        return dict(self.metadata)

    def unload(self):
        self.unloaded = True


def test_registry_exact_missing_and_ambiguous(tmp_path):
    spec = make_spec(tmp_path, "p0")
    registry = LearnedPolicyRegistry([spec])

    assert registry.resolve("put_object_in_container", "Alice", "ur5e_robotiq", "pack") == spec
    with pytest.raises(PolicyRegistryError) as missing:
        registry.resolve("PUT_OBJECT_IN_CONTAINER", "Bob", "panda", "pack")
    assert missing.value.code == "POLICY_NOT_FOUND"

    registry.register(make_spec(tmp_path, "p1"))
    with pytest.raises(PolicyRegistryError) as ambiguous:
        registry.resolve("PUT_OBJECT_IN_CONTAINER", "Alice", "ur5e_robotiq", "pack")
    assert ambiguous.value.code == "AMBIGUOUS_POLICY"


def test_registry_static_checkpoint_validation(tmp_path):
    spec = make_spec(tmp_path, "p0")
    assert LearnedPolicyRegistry([spec]).validate_static(spec) is True
    bad = LearnedPolicySpec.from_dict(dict(spec.to_dict(), checkpoint=str(tmp_path / "missing")))
    with pytest.raises(PolicyRegistryError):
        LearnedPolicyRegistry([bad]).validate_static(bad)


def test_policy_handle_cache_validates_metadata_and_unloads(tmp_path):
    spec0 = make_spec(tmp_path, "p0")
    spec1 = make_spec(tmp_path, "p1")
    handles = {
        "p0": FakeHandle(
            {
                "schema_hash": "schema",
                "action_representation": "absolute_joint_position_plus_gripper",
                "policy_type": "ACT",
                "chunk_size": 2,
            }
        ),
        "p1": FakeHandle(
            {
                "schema_hash": "schema",
                "action_representation": "absolute_joint_position_plus_gripper",
                "policy_type": "ACT",
                "chunk_size": 2,
            }
        ),
    }
    cache = BoundedPolicyHandleCache(lambda spec: handles[spec.policy_id], max_size=1)

    assert cache.get(spec0) is handles["p0"]
    assert cache.get(spec1) is handles["p1"]
    assert handles["p0"].unloaded is True

    mismatch = make_spec(tmp_path, "bad", schema_hash="expected")
    bad_cache = BoundedPolicyHandleCache(lambda spec: FakeHandle({"schema_hash": "other"}), max_size=1)
    with pytest.raises(PolicyHandleError) as exc:
        bad_cache.get(mismatch)
    assert exc.value.code == "CHECKPOINT_METADATA_MISMATCH"

