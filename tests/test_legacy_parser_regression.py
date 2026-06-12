import numpy as np
import pytest


pytest.importorskip("dm_control")

from prompting.parser import LLMResponseParser


class RobotState:
    def __init__(self):
        self.ee_pose = np.array([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.contacts = set()


class Obs:
    def __init__(self):
        self.ur5e_robotiq = RobotState()
        self.panda = RobotState()
        self.objects = {}


class Env:
    robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}


def test_legacy_action_only_wait_response_still_parses():
    parser = LLMResponseParser(
        Env(),
        "action_only",
        Env.robot_name_map,
        ["NAME", "ACTION"],
    )

    success, _, plans = parser.parse(
        Obs(),
        "EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT",
    )

    assert success
    assert len(plans) == 1


def test_legacy_action_and_path_wait_response_still_parses():
    parser = LLMResponseParser(
        Env(),
        "action_and_path",
        Env.robot_name_map,
        ["NAME", "ACTION", "PATH"],
    )
    path = "[(0.0,0.0,0.4),(0.0,0.0,0.4),(0.0,0.0,0.4),(0.0,0.0,0.4)]"

    success, _, plans = parser.parse(
        Obs(),
        "EXECUTE\nNAME Alice ACTION WAIT PATH {}\nNAME Bob ACTION WAIT PATH {}".format(path, path),
    )

    assert success
    assert len(plans) == 1
