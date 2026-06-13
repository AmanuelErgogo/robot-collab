"""Explicit ACT chunk execution queue for Phase 4."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


class ActionQueueError(ValueError):
    """Raised when a chunk/horizon queue contract is violated."""


@dataclass(frozen=True)
class QueuedAction:
    action: np.ndarray
    chunk_id: int
    chunk_offset: int
    execution_offset: int

    def trace_dict(self, env_step: int) -> Dict[str, object]:
        return {
            "env_step": int(env_step),
            "chunk_id": int(self.chunk_id),
            "chunk_offset": int(self.chunk_offset),
            "execution_offset": int(self.execution_offset),
            "action_shape": list(self.action.shape),
        }


class ACTActionQueue:
    """Execute only a configured horizon from each predicted ACT chunk."""

    def __init__(self, chunk_size: int, execution_horizon: Optional[int] = None) -> None:
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ActionQueueError("chunk_size must be positive")
        horizon = self.chunk_size if execution_horizon is None else int(execution_horizon)
        if horizon <= 0:
            raise ActionQueueError("execution_horizon must be positive")
        if horizon > self.chunk_size:
            raise ActionQueueError("execution_horizon cannot exceed chunk_size")
        self.execution_horizon = horizon
        self._items: List[QueuedAction] = []
        self._next_chunk_id = 0

    @property
    def next_chunk_id(self) -> int:
        return self._next_chunk_id

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []

    def reset(self) -> None:
        self.clear()
        self._next_chunk_id = 0

    def load_chunk(self, chunk: np.ndarray) -> int:
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ActionQueueError("action chunk must have shape [chunk, action_dim]")
        if arr.shape[0] < self.execution_horizon:
            raise ActionQueueError(
                "action chunk length {} is smaller than execution_horizon {}".format(
                    arr.shape[0], self.execution_horizon
                )
            )
        if arr.shape[0] != self.chunk_size:
            raise ActionQueueError("action chunk length {} does not match checkpoint chunk_size {}".format(arr.shape[0], self.chunk_size))
        if not np.all(np.isfinite(arr)):
            raise ActionQueueError("action chunk contains NaN or Inf")
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._items = [
            QueuedAction(
                action=np.ascontiguousarray(arr[offset], dtype=np.float32),
                chunk_id=chunk_id,
                chunk_offset=offset,
                execution_offset=offset,
            )
            for offset in range(self.execution_horizon)
        ]
        return chunk_id

    def pop(self) -> QueuedAction:
        if not self._items:
            raise ActionQueueError("action queue is empty")
        return self._items.pop(0)

