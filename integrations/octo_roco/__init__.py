"""Octo VLA integration for RoCo subtask skills.

Provides:
  OctoHandle          — LearnedPolicyHandle backed by an Octo checkpoint
  RoCoToRLDSConverter — converts RoCo NPZ demo directories → RLDS TFRecord dataset
  OctoTrainingConfig  — immutable config for Octo fine-tuning runs
  launch_octo_training — orchestrates dataset conversion + fine-tuning

Optional dependency: ``octo`` (JAX-based).  The handle falls back to a clear
ImportError with install instructions when the package is absent.
"""
