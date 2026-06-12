import types

import numpy as np

from integrations.lerobot_roco.common.errors import ErrorCode
from integrations.lerobot_roco.common.protocol import make_request
from integrations.lerobot_roco.roco_runtime.config import RoCoBridgeServerConfig
from integrations.lerobot_roco.roco_runtime.server import RoCoBridgeServer
from conftest import FakeEnv


class FakeSimAction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _server():
    config = RoCoBridgeServerConfig(image_height=8, image_width=9)
    server = RoCoBridgeServer(config=config, env=FakeEnv())
    return server


def test_ping_hello_get_spec_reset_render_digest_close():
    server = _server()
    assert server.handle_request(make_request("PING"))["ok"] is True
    assert server.handle_request(make_request("HELLO", {"supported_protocol_versions": ["0.1"]}))["ok"] is True
    spec_response = server.handle_request(make_request("GET_SPEC"))
    assert spec_response["ok"] is True
    assert spec_response["payload"]["spec"]["action"]["shape"] == [3]
    reset_response = server.handle_request(make_request("RESET", {"seed": 3, "options": {"active_agent": "Alice"}}))
    assert reset_response["ok"] is True
    assert reset_response["payload"]["info"]["is_success"] is False
    assert reset_response["payload"]["observation"]["agent_pos"].shape == (5,)
    render_response = server.handle_request(make_request("RENDER"))
    assert render_response["payload"]["image"].shape == (8, 9, 3)
    digest_response = server.handle_request(make_request("GET_STATE_DIGEST"))
    assert "qpos_sha256" in digest_response["payload"]
    assert server.handle_request(make_request("CLOSE_EPISODE"))["ok"] is True


def test_reset_accepts_seed_none():
    server = _server()
    response = server.handle_request(make_request("RESET", {"seed": None, "options": {"active_agent": "Alice"}}))
    assert response["ok"] is True
    assert response["payload"]["info"]["seed"] == 0


def test_step_validates_episode_and_returns_transition():
    server = _server()
    reset = server.handle_request(make_request("RESET", {"seed": 0, "options": {"active_agent": "Alice"}}))["payload"]

    def fake_to_sim_action(action):
        kwargs, info = server.action_adapter.build_sim_action_kwargs(action)
        return FakeSimAction(**kwargs), info

    server.action_adapter.to_sim_action = fake_to_sim_action
    response = server.handle_request(
        make_request(
            "STEP",
            {
                "episode_id": reset["episode_id"],
                "step_index": reset["step_index"],
                "action": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
        )
    )
    assert response["ok"] is True
    assert response["payload"]["step_index"] == 1
    assert response["payload"]["info"]["is_success"] is False


def test_stale_step_index_is_rejected():
    server = _server()
    reset = server.handle_request(make_request("RESET", {"seed": 0, "options": {"active_agent": "Alice"}}))["payload"]
    response = server.handle_request(
        make_request(
            "STEP",
            {
                "episode_id": reset["episode_id"],
                "step_index": 99,
                "action": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
        )
    )
    assert response["ok"] is False
    assert response["error"]["code"] == ErrorCode.STEP_INDEX_MISMATCH
