"""Learned skill progress and failure monitors."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from .models import MonitorEvent, ProgressStage


EVENT_TIMEOUT = "TIMEOUT"
EVENT_NONFINITE_ACTION = "NONFINITE_ACTION"
EVENT_ACTION_OUT_OF_BOUNDS = "ACTION_OUT_OF_BOUNDS"
EVENT_NO_PROGRESS = "NO_PROGRESS"
EVENT_BRIDGE_FAILURE = "BRIDGE_FAILURE"
EVENT_POLICY_INFERENCE_FAILURE = "POLICY_INFERENCE_FAILURE"
EVENT_MANUAL_INTERRUPT = "MANUAL_INTERRUPT"
EVENT_OBJECT_LOST = "OBJECT_LOST"
EVENT_TARGET_OCCUPIED = "TARGET_OCCUPIED"
EVENT_COLLISION_RISK = "COLLISION_RISK"
EVENT_MISSED_GRASP = "MISSED_GRASP"
EVENT_SLIPPAGE = "SLIPPAGE"


def validate_action(action, low, high, tolerance=0.0):
    arr = np.asarray(action, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return MonitorEvent(
            EVENT_NONFINITE_ACTION,
            "error",
            "Policy produced NaN or Inf action.",
            {"shape": list(arr.shape)},
            "Stop learned execution.",
        )
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    below = int(np.sum(arr < low_arr - float(tolerance)))
    above = int(np.sum(arr > high_arr + float(tolerance)))
    if below or above:
        return MonitorEvent(
            EVENT_ACTION_OUT_OF_BOUNDS,
            "error",
            "Policy action is outside native action bounds.",
            {"below_low": below, "above_high": above},
            "Request fallback or abort.",
        )
    return None


@dataclass
class NoProgressMonitor:
    window: int = 20
    patience: int = 3
    state_epsilon: float = 1e-4
    action_epsilon: float = 1e-4
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    stalled_windows: int = 0

    def reset(self):
        self.states = []
        self.actions = []
        self.stalled_windows = 0

    def observe(self, state, action):
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        if len(self.states) < self.window:
            return None
        self.states = self.states[-self.window :]
        self.actions = self.actions[-self.window :]
        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        state_span = float(np.max(np.linalg.norm(states - states[0], axis=1)))
        action_span = float(np.max(np.linalg.norm(actions - actions[0], axis=1)))
        if state_span <= self.state_epsilon and action_span <= self.action_epsilon:
            self.stalled_windows += 1
        else:
            self.stalled_windows = 0
        if self.stalled_windows >= self.patience:
            return MonitorEvent(
                EVENT_NO_PROGRESS,
                "warning",
                "No measurable state or action progress over the configured window.",
                {"state_span": state_span, "action_span": action_span, "window": self.window},
                "Caller may request RRT fallback.",
            )
        return None


def infer_progress_stage(env, obs, agent_name, object_name, target_name):
    held = None
    if hasattr(env, "get_agent_held_object"):
        held = env.get_agent_held_object(obs, agent_name)
    packed_slot = None
    if hasattr(env, "get_packed_slot_for_object"):
        packed_slot = env.get_packed_slot_for_object(obs, object_name)
    if packed_slot == target_name:
        return ProgressStage.STABLE
    if held == object_name:
        return ProgressStage.GRASPED
    return ProgressStage.NOT_STARTED


def monitor_info_events(info):
    events = []
    info = dict(info or {})
    if info.get("object_lost"):
        events.append(MonitorEvent(EVENT_OBJECT_LOST, "error", "Object was reported lost.", dict(info), "Abort."))
    if info.get("target_occupied"):
        events.append(MonitorEvent(EVENT_TARGET_OCCUPIED, "error", "Target was reported occupied.", dict(info), "Abort."))
    if info.get("collision") or info.get("contact_violation"):
        events.append(MonitorEvent(EVENT_COLLISION_RISK, "warning", "Collision/contact risk was reported.", dict(info), "Abort or fallback."))
    if info.get("slip_detected"):
        events.append(MonitorEvent(EVENT_SLIPPAGE, "warning", "Object slippage was reported.", dict(info), "Fallback may recover."))
    return events

