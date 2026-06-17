"""Benchmark observation processor.

This module only maps raw RoCo bridge observations to stable benchmark keys.
It intentionally avoids normalization, batching, tokenization, and device moves.
Those belong in policy-specific processors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


class BenchmarkProcessorError(ValueError):
    """Raised when a raw observation violates the benchmark processor contract."""


@dataclass(frozen=True)
class ProcessorTrace:
    raw_keys: Tuple[str, ...]
    camera_shapes: Mapping[str, Tuple[int, ...]]
    output_shapes: Mapping[str, Tuple[int, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_keys": list(self.raw_keys),
            "camera_shapes": {key: list(value) for key, value in self.camera_shapes.items()},
            "output_shapes": {key: list(value) for key, value in self.output_shapes.items()},
        }


@dataclass
class BenchmarkObservationProcessor:
    required_cameras: Tuple[str, ...] = ("front", "active_agent")
    last_trace: Optional[ProcessorTrace] = field(default=None, init=False)

    def process(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        if "agent_pos" not in observation:
            raise BenchmarkProcessorError("raw observation missing agent_pos")
        if "pixels" not in observation:
            raise BenchmarkProcessorError("raw observation missing pixels")
        pixels = observation["pixels"]
        if not isinstance(pixels, Mapping):
            raise BenchmarkProcessorError("pixels must be a camera mapping")

        state = np.asarray(observation["agent_pos"], dtype=np.float32)
        if state.ndim != 1:
            raise BenchmarkProcessorError("agent_pos must be a rank-1 state vector")
        if not np.all(np.isfinite(state)):
            raise BenchmarkProcessorError("agent_pos contains NaN or Inf")

        processed: Dict[str, Any] = {
            "observation.state": np.ascontiguousarray(state, dtype=np.float32),
        }
        camera_shapes: Dict[str, Tuple[int, ...]] = {}
        output_shapes: Dict[str, Tuple[int, ...]] = {"observation.state": tuple(state.shape)}
        for camera in self.required_cameras:
            if camera not in pixels:
                raise BenchmarkProcessorError("missing required camera '{}'".format(camera))
            image = np.asarray(pixels[camera])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise BenchmarkProcessorError("{} must be HWC RGB, got {}".format(camera, image.shape))
            if image.dtype != np.uint8:
                raise BenchmarkProcessorError("{} must be uint8, got {}".format(camera, image.dtype))
            key = "observation.images.{}".format(camera)
            processed[key] = np.ascontiguousarray(image, dtype=np.uint8)
            camera_shapes[camera] = tuple(image.shape)
            output_shapes[key] = tuple(image.shape)

        self.last_trace = ProcessorTrace(
            raw_keys=tuple(sorted(str(key) for key in observation.keys())),
            camera_shapes=camera_shapes,
            output_shapes=output_shapes,
        )
        return processed
