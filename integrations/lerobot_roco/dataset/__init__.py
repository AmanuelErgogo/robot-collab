"""Phase 2 RoCo expert dataset pipeline.

The package is intentionally split from the Gymnasium client and from normal
RoCo modules. Simulator-specific imports live behind functions that run in the
RoCo runtime; LeRobot imports live behind writer/validator compatibility calls.
"""

from .schema import FeatureSpec, SkillDataSchema, build_schema_from_env_spec
from .episode_sampler import VariationSpec, DeterministicVariationSampler
from .recorder import EpisodeRecord, TransitionFrame, RocoTransitionObserver
from .writer import AtomicEpisodeWriter
from .validator import ValidationIssue, ValidationReport, validate_dataset

__all__ = [
    "AtomicEpisodeWriter",
    "DeterministicVariationSampler",
    "EpisodeRecord",
    "FeatureSpec",
    "RocoTransitionObserver",
    "SkillDataSchema",
    "TransitionFrame",
    "ValidationIssue",
    "ValidationReport",
    "VariationSpec",
    "build_schema_from_env_spec",
    "validate_dataset",
]
