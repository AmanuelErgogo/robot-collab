"""Terminal human-interface tests (plan test 6)."""

from rocobench.crie_bt.pipeline.communication import (
    ScriptedHumanCommunicationInterface,
    TerminalCommunicationInterface,
    parse_human_response,
)
from rocobench.crie_bt.pipeline.interfaces import HumanInstruction


def test_terminal_human_interface_returns_accept_reject_counter_done():
    cases = {
        "1": "accept", "accept": "accept", "yes": "accept",
        "2": "reject", "reject": "reject", "no": "reject",
        "3": "counter_propose", "counter use bottle B": "counter_propose",
        "done": "done", "finished": "done",
    }
    for text, expected in cases.items():
        assert parse_human_response(text).action == expected


def test_unknown_response_is_unknown():
    assert parse_human_response("banana").action == "unknown"
    assert parse_human_response("").action == "unknown"


def test_terminal_interface_uses_stdin(monkeypatch):
    comm = TerminalCommunicationInterface()
    comm.send_to_human(HumanInstruction(text="Please grasp the bottle.", target_human_id="human_1"))
    monkeypatch.setattr("builtins.input", lambda *a, **k: "2")
    response = comm.read_human_response()
    assert response.action == "reject"
    # The transcript captures both directions.
    roles = [entry["role"] for entry in comm.transcript]
    assert "robot_to_human" in roles and "human_to_robot" in roles


def test_scripted_interface_replays_queue_then_defaults():
    comm = ScriptedHumanCommunicationInterface(responses=["accept", "3"], default_action="done")
    assert comm.read_human_response().action == "accept"
    assert comm.read_human_response().action == "counter_propose"
    # Queue exhausted -> default keeps the episode from blocking.
    assert comm.read_human_response().action == "done"
