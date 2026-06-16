"""Parser for single-step MetaWorld control outputs."""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np


ACTION_PATTERN = re.compile(
    r"ACTION\s*\[\s*([^\]]+)\s*\]",
    flags=re.IGNORECASE | re.MULTILINE,
)


class MetaWorldActionParser:
    """Parse direct 4D Sawyer control deltas from an LLM response."""

    def parse(self, response):
        if "EXECUTE" not in response:
            return False, "Response missing EXECUTE marker.", None

        match = ACTION_PATTERN.search(response)
        if match is None:
            return False, "Response missing ACTION [dx, dy, dz, gripper] block.", None

        raw_values = [
            value.strip() for value in re.split(r"[\s,]+", match.group(1).strip()) if value.strip()
        ]
        if len(raw_values) != 4:
            return False, "Expected exactly 4 action values, got {}.".format(len(raw_values)), None

        try:
            action = np.array([float(value) for value in raw_values], dtype=np.float32)
        except ValueError as exc:
            return False, "Action values must be floats: {}".format(exc), None

        clipped = np.clip(action, -1.0, 1.0)
        if not np.allclose(action, clipped):
            reason = "Action values were clipped into [-1, 1]."
        else:
            reason = ""
        return True, reason, clipped

    @staticmethod
    def format_action(action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        return "EXECUTE\nACTION [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*action.tolist())
