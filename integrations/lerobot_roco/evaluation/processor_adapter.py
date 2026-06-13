"""RoCo raw observation to LeRobot policy-input adapter."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


class ProcessorContractError(ValueError):
    """Raised when raw observations do not match the policy feature contract."""


def _shape_tuple(value: Any) -> Tuple[int, ...]:
    return tuple(int(x) for x in value)


def _feature_shape(feature: Any) -> Tuple[int, ...]:
    if isinstance(feature, Mapping):
        return _shape_tuple(feature.get("shape", ()))
    return _shape_tuple(getattr(feature, "shape"))


def _feature_type(feature: Any) -> str:
    if isinstance(feature, Mapping):
        return str(feature.get("type", ""))
    return str(getattr(feature, "type", ""))


@dataclass(frozen=True)
class ProcessorTrace:
    raw_keys: Tuple[str, ...]
    processed_keys: Tuple[str, ...]
    raw_camera_shapes: Mapping[str, Tuple[int, ...]]
    processed_shapes: Mapping[str, Tuple[int, ...]]
    state_shape: Tuple[int, ...]
    instruction_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_keys": list(self.raw_keys),
            "processed_keys": list(self.processed_keys),
            "raw_camera_shapes": {key: list(value) for key, value in self.raw_camera_shapes.items()},
            "processed_shapes": {key: list(value) for key, value in self.processed_shapes.items()},
            "state_shape": list(self.state_shape),
            "instruction_key": self.instruction_key,
        }


@dataclass
class RoCoPolicyObservationAdapter:
    """Map Phase 0 raw Gym observations to LeRobot policy feature keys.

    The raw bridge contract remains ``pixels`` and ``agent_pos``. This adapter
    only performs key mapping and image HWC uint8 to CHW float32 [0, 1]
    conversion. Normalization, batching, device moves, and tokenization remain
    the saved LeRobot policy processor's responsibility.
    """

    input_features: Mapping[str, Any]
    task_instruction: Optional[str] = None
    prefer_torch: bool = True
    last_trace: Optional[ProcessorTrace] = field(default=None, init=False)

    def _expected_state_shape(self) -> Tuple[int, ...]:
        feature = self.input_features.get("observation.state")
        if feature is None:
            raise ProcessorContractError("policy does not declare observation.state")
        return _feature_shape(feature)

    def _image_feature_keys(self) -> Tuple[str, ...]:
        keys = []
        for key, feature in self.input_features.items():
            if key.startswith("observation.images.") or "VISUAL" in _feature_type(feature):
                keys.append(str(key))
        return tuple(sorted(keys))

    def _instruction_keys(self) -> Tuple[str, ...]:
        keys = []
        for key in self.input_features.keys():
            lowered = str(key).lower()
            if "task" in lowered or "instruction" in lowered or "language" in lowered:
                keys.append(str(key))
        return tuple(sorted(keys))

    def _to_policy_value(self, array: np.ndarray) -> Any:
        if self.prefer_torch:
            try:
                import torch

                return torch.as_tensor(array, dtype=torch.float32)
            except Exception:
                pass
        return array

    def _convert_image(self, key: str, raw_pixels: Mapping[str, Any]) -> Any:
        alias = key.rsplit(".", 1)[-1]
        if alias not in raw_pixels:
            raise ProcessorContractError("raw observation missing camera alias '{}' for {}".format(alias, key))
        arr = np.asarray(raw_pixels[alias])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ProcessorContractError("{} must be HWC RGB, got {}".format(alias, arr.shape))
        if arr.dtype != np.uint8:
            raise ProcessorContractError("{} must be uint8, got {}".format(alias, arr.dtype))
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
        expected = _feature_shape(self.input_features[key])
        if tuple(chw.shape) != expected:
            raise ProcessorContractError("{} expected CHW {}, got {}".format(key, expected, chw.shape))
        return self._to_policy_value(np.ascontiguousarray(chw, dtype=np.float32))

    def to_policy_inputs(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        if "pixels" not in observation or "agent_pos" not in observation:
            raise ProcessorContractError("raw observation must include pixels and agent_pos")
        pixels = observation["pixels"]
        if not isinstance(pixels, Mapping):
            raise ProcessorContractError("raw pixels must be a camera mapping")
        state = np.asarray(observation["agent_pos"], dtype=np.float32)
        expected_state_shape = self._expected_state_shape()
        if tuple(state.shape) != expected_state_shape:
            raise ProcessorContractError("observation.state expected {}, got {}".format(expected_state_shape, state.shape))
        if not np.all(np.isfinite(state)):
            raise ProcessorContractError("observation.state contains NaN or Inf")

        policy_inputs: Dict[str, Any] = {
            "observation.state": self._to_policy_value(np.ascontiguousarray(state, dtype=np.float32)),
        }
        raw_camera_shapes: Dict[str, Tuple[int, ...]] = {}
        processed_shapes: Dict[str, Tuple[int, ...]] = {"observation.state": tuple(state.shape)}
        for key in self._image_feature_keys():
            alias = key.rsplit(".", 1)[-1]
            raw_camera_shapes[alias] = tuple(np.asarray(pixels[alias]).shape) if alias in pixels else ()
            converted = self._convert_image(key, pixels)
            policy_inputs[key] = converted
            processed_shapes[key] = tuple(converted.shape)

        instruction_keys = self._instruction_keys()
        if instruction_keys:
            if self.task_instruction is None:
                raise ProcessorContractError("policy expects instruction/task key but task_instruction is unset")
            for key in instruction_keys:
                policy_inputs[key] = self.task_instruction

        self.last_trace = ProcessorTrace(
            raw_keys=tuple(sorted(str(k) for k in observation.keys())),
            processed_keys=tuple(sorted(policy_inputs.keys())),
            raw_camera_shapes=raw_camera_shapes,
            processed_shapes=processed_shapes,
            state_shape=tuple(state.shape),
            instruction_key=instruction_keys[0] if instruction_keys else None,
        )
        return policy_inputs
