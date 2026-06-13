"""Artifact writer for learned skill execution."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = "{}.tmp".format(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def append_jsonl(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":")))
        f.write("\n")


@dataclass
class LearnedSkillArtifactWriter:
    artifact_dir: Optional[str]
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    chunk_ids: list = field(default_factory=list)
    chunk_offsets: list = field(default_factory=list)

    def __post_init__(self):
        if self.artifact_dir:
            os.makedirs(self.artifact_dir, exist_ok=True)

    def path(self, filename):
        if not self.artifact_dir:
            return None
        return os.path.join(self.artifact_dir, filename)

    def write_json(self, filename, data):
        path = self.path(filename)
        if path:
            write_json(path, data)

    def append_jsonl(self, filename, data):
        path = self.path(filename)
        if path:
            append_jsonl(path, data)

    def record_step(self, state, action, chunk_id, chunk_offset):
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.chunk_ids.append(int(chunk_id))
        self.chunk_offsets.append(int(chunk_offset))

    def finalize_arrays(self):
        if not self.artifact_dir:
            return
        np.savez_compressed(
            self.path("actions.npz"),
            actions=np.asarray(self.actions, dtype=np.float32),
            chunk_ids=np.asarray(self.chunk_ids, dtype=np.int32),
            chunk_offsets=np.asarray(self.chunk_offsets, dtype=np.int32),
        )
        np.savez_compressed(
            self.path("states.npz"),
            states=np.asarray(self.states, dtype=np.float32),
        )

