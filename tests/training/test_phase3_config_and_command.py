import os

from integrations.lerobot_roco.training.compatibility import build_lerobot_train_command
from integrations.lerobot_roco.training.config import load_training_config


def test_debug_config_loads_and_builds_lerobot_command():
    config = load_training_config("configs/training/act_pack_put_debug.yaml")

    command = build_lerobot_train_command(config, output_dir="/tmp/roco-act-debug")

    assert command[0] == "lerobot-train"
    assert "--policy.type=act" in command
    assert "--dataset.repo_id=local/roco-pack-put-object-debug" in command
    assert "--dataset.root=artifacts/datasets/pack_put_object_debug_lerobot" in command
    assert "--steps=50" in command
    assert "--batch_size=2" in command
    assert "--dataset.image_transforms.enable=false" in command
    assert "--policy.push_to_hub=false" in command
    assert "--output_dir=/tmp/roco-act-debug" in command


def test_overfit_config_disables_augmentation_and_selects_episode_zero():
    config = load_training_config("configs/training/act_pack_put_overfit.yaml")
    command = build_lerobot_train_command(config, output_dir=os.devnull)

    assert config.episodes == (0,)
    assert config.image_transforms_enable is False
    assert config.use_imagenet_stats is False
    assert "--dataset.episodes=[0]" in command
