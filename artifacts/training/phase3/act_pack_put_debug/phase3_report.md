# Phase 3 ACT Training Report

## Run
- run_id: `act_pack_put_debug`
- status: `dry_run`
- phase: `phase3_act_training`
- resolved_config_hash: `eaee87cf7c071a0568fce2eeca88a6011d71a2459bf9527f6f7e6914921e48eb`

## Dataset
- dataset_root: `/home/amanu/robot-collab/artifacts/datasets/pack_put_object_debug`
- dataset_revision: `7e440178ae8d`
- schema_hash: `7e440178ae8debeb396a14bbd98531382da78fd01dffa3d41b84ba6f61903d2a`
- bridge_protocol: `0.1`
- action_representation: `absolute_joint_position_plus_gripper`
- fps: `5.0`

## Preflight
- ok: `True`
- issues: `none`

## LeRobot Environment
- ok: `True`
- lerobot_version: `0.3.4`
- lerobot_commit: `0878c6880fa4fbadf0742751cf7b015f2d63a769`

## Command
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/roco-pack-put-object-debug \
  --dataset.root=artifacts/datasets/pack_put_object_debug_lerobot \
  --output_dir=/home/amanu/robot-collab/artifacts/training/phase3/act_pack_put_debug/lerobot_output \
  --steps=50 \
  --batch_size=2 \
  --num_workers=0 \
  --eval_freq=0 \
  --save_freq=25 \
  --log_freq=10 \
  --seed=1000 \
  --policy.device=cpu \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --dataset.use_imagenet_stats=false \
  --dataset.image_transforms.enable=false \
  --wandb.enable=false
```

## Notes
- Offline training diagnostics are not manipulation success.
- Closed-loop simulator rollout is Phase 4.
