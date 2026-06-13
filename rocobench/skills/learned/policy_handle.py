"""Policy handle abstractions and bounded cache.

No LeRobot imports are allowed here. Real policy loading should be supplied by a
client-side adapter that implements this handle interface.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import PolicyHandleError


@dataclass(frozen=True)
class NativeActionChunk:
    actions: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self):
        arr = np.asarray(self.actions, dtype=np.float32)
        object.__setattr__(self, "actions", np.ascontiguousarray(arr, dtype=np.float32))

    def to_dict(self):
        return {
            "shape": list(self.actions.shape),
            "dtype": str(self.actions.dtype),
            "metadata": dict(self.metadata),
        }


class LearnedPolicyHandle(object):
    """Minimal policy lifecycle expected by LearnedSkillExecutor."""

    def reset(self):
        raise NotImplementedError

    def predict_native_chunk(self, observation, instruction, action_low, action_high):
        raise NotImplementedError

    def health_check(self, spec):
        return {}

    def unload(self):
        pass


def _check_metadata(spec, metadata):
    metadata = dict(metadata or {})
    checks = (
        ("schema_hash", spec.schema_hash),
        ("action_representation", spec.action_representation),
        ("policy_type", spec.policy_type),
    )
    for key, expected in checks:
        if key in metadata and str(metadata[key]).lower() != str(expected).lower():
            raise PolicyHandleError(
                "CHECKPOINT_METADATA_MISMATCH",
                "{} mismatch for {}: expected {}, got {}".format(key, spec.policy_id, expected, metadata[key]),
                evidence={"key": key, "expected": expected, "received": metadata[key]},
            )
    if "chunk_size" in metadata and int(metadata["chunk_size"]) < int(spec.execution_horizon):
        raise PolicyHandleError(
            "CHECKPOINT_HORIZON_MISMATCH",
            "execution_horizon exceeds checkpoint chunk_size.",
            evidence={"execution_horizon": spec.execution_horizon, "chunk_size": metadata["chunk_size"]},
        )


class BoundedPolicyHandleCache(object):
    def __init__(self, loader, max_size=1):
        self.loader = loader  # type: Callable[[Any], LearnedPolicyHandle]
        self.max_size = int(max_size)
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self._cache)

    def get(self, spec):
        key = spec.policy_id
        if key in self._cache:
            handle = self._cache.pop(key)
            self._cache[key] = handle
            return handle
        handle = self.loader(spec)
        metadata = handle.health_check(spec)
        _check_metadata(spec, metadata)
        while len(self._cache) >= self.max_size:
            _, old = self._cache.popitem(last=False)
            old.unload()
        self._cache[key] = handle
        return handle

    def unload(self, policy_id=None):
        if policy_id is None:
            items = list(self._cache.items())
            self._cache.clear()
            for _, handle in items:
                handle.unload()
            return
        handle = self._cache.pop(str(policy_id), None)
        if handle is not None:
            handle.unload()

    def health_check(self, spec):
        handle = self.get(spec)
        return handle.health_check(spec)

