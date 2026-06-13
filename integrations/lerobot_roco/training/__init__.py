"""Phase 3 ACT training support for RoCo x LeRobot.

This package is deliberately import-safe in the RoCo runtime.  LeRobot and
Torch imports are isolated behind adapter functions that are only called by the
Python 3.12+ training environment.
"""

from .config import Phase3TrainingConfig, load_training_config
from .feature_contract import PolicyFeatureContract, run_feature_preflight

__all__ = [
    "Phase3TrainingConfig",
    "PolicyFeatureContract",
    "load_training_config",
    "run_feature_preflight",
]
