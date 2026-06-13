from integrations.lerobot_roco.dataset.episode_sampler import (
    DeterministicVariationSampler,
    VariationSpec,
)
from integrations.lerobot_roco.dataset.schema import default_schema_for_tests


def test_schema_hash_is_stable_for_same_contract():
    schema_a = default_schema_for_tests()
    schema_b = default_schema_for_tests()

    assert schema_a.schema_hash == schema_b.schema_hash
    assert "observation.images.front" in schema_a.to_lerobot_features()
    assert schema_a.to_lerobot_features()["action"]["shape"] == (2,)


def test_variation_id_depends_on_canonical_spec_not_object_identity():
    variation = VariationSpec(
        seed=123,
        variation_index=0,
        object_name="apple",
        target_name="bin_front_left",
    )
    round_tripped = VariationSpec.from_dict(variation.to_dict())

    assert variation.variation_id == round_tripped.variation_id


def test_sampler_is_deterministic():
    sampler_a = DeterministicVariationSampler(42, object_names=["apple", "banana"], target_names=["slot_a", "slot_b"])
    sampler_b = DeterministicVariationSampler(42, object_names=["apple", "banana"], target_names=["slot_a", "slot_b"])

    assert [sampler_a.sample(i).to_dict() for i in range(5)] == [sampler_b.sample(i).to_dict() for i in range(5)]
