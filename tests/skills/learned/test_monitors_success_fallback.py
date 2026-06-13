import numpy as np

from rocobench.skills.learned.fallback import FallbackConfig, FallbackController
from rocobench.skills.learned.monitors import EVENT_ACTION_OUT_OF_BOUNDS, EVENT_NONFINITE_ACTION, NoProgressMonitor, validate_action
from rocobench.skills.learned.success import StableSkillSuccessChecker


class FakeEnv:
    def __init__(self):
        self.held = None
        self.packed_slot = None
        self.occupancy = {"bin_front_left": None}

    def get_agent_held_object(self, obs, agent_name):
        return self.held

    def get_packed_slot_for_object(self, obs, object_name):
        return self.packed_slot

    def get_slot_occupancy(self, obs):
        return dict(self.occupancy)


def test_action_monitors_positive_and_negative():
    assert validate_action([0.0, 1.0], [-1.0, 0.0], [1.0, 2.0]) is None
    assert validate_action([float("nan"), 1.0], [-1.0, 0.0], [1.0, 2.0]).code == EVENT_NONFINITE_ACTION
    assert validate_action([2.0, 1.0], [-1.0, 0.0], [1.0, 2.0]).code == EVENT_ACTION_OUT_OF_BOUNDS


def test_no_progress_monitor_is_conservative():
    monitor = NoProgressMonitor(window=3, patience=2, state_epsilon=1e-6, action_epsilon=1e-6)
    assert monitor.observe([0.0], [0.0]) is None
    assert monitor.observe([0.0], [0.0]) is None
    assert monitor.observe([0.0], [0.0]) is None
    assert monitor.observe([0.0], [0.0]).code == "NO_PROGRESS"


def test_stable_success_requires_target_release_and_stability():
    env = FakeEnv()
    checker = StableSkillSuccessChecker(stable_checks=2)
    assert checker.check(env, object(), "Alice", "apple", "bin_front_left") is False

    env.packed_slot = "bin_front_left"
    env.occupancy["bin_front_left"] = "apple"
    env.held = "apple"
    assert checker.check(env, object(), "Alice", "apple", "bin_front_left") is False

    env.held = None
    assert checker.check(env, object(), "Alice", "apple", "bin_front_left") is False
    assert checker.check(env, object(), "Alice", "apple", "bin_front_left") is True


def test_fallback_return_request_and_automatic_separation():
    config = FallbackConfig(mode="return_request", allowed_reasons=("NO_PROGRESS",))
    decision = FallbackController(config).decide("NO_PROGRESS")
    assert decision.recommended is True
    assert decision.used is False
    assert decision.success is False

