"""ZeroMQ request/reply server for the RoCo bridge."""

import hashlib
import logging
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from integrations.lerobot_roco.common.errors import (
    ErrorCode,
    RoCoBridgeError,
    RoCoEpisodeError,
    RoCoProtocolError,
)
from integrations.lerobot_roco.common.protocol import (
    PROTOCOL_VERSION,
    make_error_response,
    make_exception_response,
    make_success_response,
    ping_payload,
    validate_request_envelope,
)
from integrations.lerobot_roco.common.serialization import pack_message, unpack_message
from integrations.lerobot_roco.common.types import ArraySpec, CameraSpec, RoCoEnvSpec
from .action_adapter import RoCoActionAdapter, decode_sim_action
from .config import RoCoBridgeServerConfig
from .env_factory import create_roco_env
from .episode import EpisodeStateMachine, EpisodeStatus
from .observation_adapter import RoCoObservationAdapter

LOGGER = logging.getLogger(__name__)


def _git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def _digest_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


class RoCoBridgeServer:
    def __init__(
        self,
        config: RoCoBridgeServerConfig,
        env: Optional[Any] = None,
    ) -> None:
        self.config = config
        logging.basicConfig(level=getattr(logging, config.request_log_level.upper(), logging.INFO))
        self.env = env if env is not None else create_roco_env(config)
        self.episode = EpisodeStateMachine()
        self.episode.mark_ready()
        self._running = False
        self._live_viewer = None
        if self.config.live_view:
            from .live_view import LiveViewer

            self._live_viewer = LiveViewer(camera_alias=self.config.live_view_camera)
        self._rebuild_adapters()

    def _rebuild_adapters(self) -> None:
        self.action_adapter = RoCoActionAdapter(self.env, self.config.active_agent)
        self.observation_adapter = RoCoObservationAdapter(
            self.env,
            self.config.active_agent,
            self.config.camera_aliases or {},
            self.config.image_height,
            self.config.image_width,
        )
        self.spec = self._build_spec()

    def _show_live_observation(self, observation: Dict[str, Any]) -> None:
        if self._live_viewer is None:
            return
        self._live_viewer.show_observation(observation)

    def _show_live_image(self, image: Any) -> None:
        if self._live_viewer is None:
            return
        self._live_viewer.show_image(image)

    def _physics_timestep(self) -> float:
        try:
            return float(self.env.physics.timestep())
        except Exception:
            return 0.0

    def _effective_fps(self) -> float:
        timestep = self._physics_timestep()
        sim_forward_steps = float(getattr(self.env, "sim_forward_steps", 1))
        duration = timestep * sim_forward_steps
        return 1.0 / duration if duration > 0 else 0.0

    def _build_spec(self) -> RoCoEnvSpec:
        action_layout = self.action_adapter.layout
        obs_layout = self.observation_adapter.layout
        cameras = tuple(
            CameraSpec(
                name=alias,
                height=self.config.image_height,
                width=self.config.image_width,
                channels=3,
                dtype="uint8",
            )
            for alias in sorted(obs_layout.camera_aliases.keys())
        )
        passive_agents = tuple(agent for agent in sorted(self.env.robots.keys()) if agent != self.config.active_agent)
        metadata = {
            "robot_model": action_layout.active_robot_name,
            "agent_to_robot": dict(getattr(self.env, "robot_name_map_inv", {})),
            "joint_names": list(action_layout.joint_names),
            "joint_ctrl_indices": list(action_layout.joint_ctrl_indices),
            "joint_qpos_indices": list(action_layout.joint_qpos_indices),
            "joint_qvel_indices": list(obs_layout.joint_qvel_indices),
            "gripper_ctrl_index": action_layout.gripper_ctrl_index,
            "gripper_name": action_layout.gripper_name,
            "state_field_names": list(obs_layout.state_field_names),
            "action_field_names": list(action_layout.field_names),
            "action_control_ranges": {
                "low": action_layout.low.tolist(),
                "high": action_layout.high.tolist(),
            },
            "camera_aliases": dict(obs_layout.camera_aliases),
            "camera_names": list(getattr(self.env, "render_cameras", [])),
            "sim_forward_steps": int(getattr(self.env, "sim_forward_steps", 0)),
            "render_freq": int(getattr(self.env, "render_freq", 0)),
            "sim_save_freq": int(getattr(self.env, "sim_save_freq", 0)),
            "physics_timestep": self._physics_timestep(),
            "effective_env_step_duration": (1.0 / self._effective_fps()) if self._effective_fps() > 0 else 0.0,
            "randomization_enabled": bool(getattr(self.env, "randomize_init", False)),
            "reward_success_description": "PackGroceryTask returns reward=1 and done=True when every grocery item is in or aligned with the bin.",
            "roco_git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
        }
        return RoCoEnvSpec(
            protocol_version=PROTOCOL_VERSION,
            task="pack",
            task_description="Pack grocery items into the grocery bin.",
            active_agent=self.config.active_agent,
            passive_agents=passive_agents,
            max_episode_steps=self.config.max_episode_steps,
            effective_fps=self._effective_fps(),
            cameras=cameras,
            observation_state=ArraySpec(
                name="agent_pos",
                shape=(obs_layout.state_dim,),
                dtype="float32",
                field_names=obs_layout.state_field_names,
            ),
            action=ArraySpec(
                name="action",
                shape=(action_layout.action_dim,),
                dtype="float32",
                low=[float(x) for x in action_layout.low],
                high=[float(x) for x in action_layout.high],
                field_names=action_layout.field_names,
            ),
            action_mode="absolute_joint_position_plus_gripper",
            success_semantics="is_success is bool(done or reward > 0) using PackGroceryTask.get_reward_done.",
            metadata=metadata,
        )

    def _success_info(self, obs: Any, reward: Optional[float] = None, done: Optional[bool] = None) -> Dict[str, Any]:
        if reward is None or done is None:
            if hasattr(self.env, "get_reward_done"):
                try:
                    reward, done = self.env.get_reward_done(obs)
                except Exception:
                    reward, done = 0.0, False
            else:
                reward, done = 0.0, False
        is_success = bool(done or float(reward) > 0)
        return {"is_success": is_success, "reward": float(reward), "done": bool(done)}

    def _seed_env(self, seed: int) -> None:
        if hasattr(self.env, "seed"):
            self.env.seed(np_seed=int(seed))

    def _reset_env(self) -> Any:
        try:
            return self.env.reset(reload=True)
        except TypeError:
            return self.env.reset()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            validate_request_envelope(
                request,
                max_payload_bytes=self.config.max_payload_bytes,
                allow_debug_commands=self.config.enable_debug_commands,
            )
            command = request["command"]
            payload = request.get("payload", {})
            handler = getattr(self, "_handle_%s" % command.lower(), None)
            if handler is None:
                raise RoCoProtocolError("Unknown command.", code=ErrorCode.UNKNOWN_COMMAND)
            response = handler(request, payload)
            LOGGER.debug(
                "bridge request command=%s request_id=%s latency_ms=%.2f",
                command,
                request["request_id"],
                (time.time() - start) * 1000.0,
            )
            return response
        except Exception as exc:
            if not isinstance(exc, RoCoBridgeError):
                LOGGER.error("Unhandled bridge error:\n%s", traceback.format_exc())
            return make_exception_response(request, exc)

    def _handle_ping(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        return make_success_response(request, ping_payload())

    def _handle_hello(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        supported = payload.get("supported_protocol_versions", [PROTOCOL_VERSION])
        if PROTOCOL_VERSION not in supported:
            raise RoCoProtocolError(
                "No supported protocol version in common.",
                code=ErrorCode.UNSUPPORTED_PROTOCOL,
                details={"supported_by_server": [PROTOCOL_VERSION], "supported_by_client": supported},
            )
        return make_success_response(
            request,
            {
                "selected_protocol_version": PROTOCOL_VERSION,
                "server_name": "roco-bridge",
                "python_version": sys.version.split()[0],
                "task": self.config.task,
                "capabilities": ["reset", "step", "render", "state_digest"],
            },
        )

    def _handle_get_spec(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        return make_success_response(request, {"spec": self.spec.to_dict()})

    def _handle_reset(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.episode.state == EpisodeStatus.EPISODE_ACTIVE:
            raise RoCoEpisodeError(
                "Episode is already active.",
                code=ErrorCode.EPISODE_ALREADY_ACTIVE,
                details={"episode_id": self.episode.episode_id, "state": self.episode.state},
            )
        options = payload.get("options", {})
        requested_agent = options.get("active_agent", self.config.active_agent)
        if requested_agent != self.config.active_agent:
            raise RoCoProtocolError(
                "Server active agent does not match reset option.",
                code=ErrorCode.UNKNOWN_AGENT,
                details={"server_active_agent": self.config.active_agent, "requested_agent": requested_agent},
            )
        seed_value = payload.get("seed", self.config.seed)
        seed = self.config.seed if seed_value is None else int(seed_value)
        self._seed_env(seed)
        obs = self._reset_env()
        self._rebuild_adapters()
        self.action_adapter.refresh_holds()
        episode_id = self.episode.reset()
        formatted = self.observation_adapter.format(obs)
        self._show_live_observation(formatted)
        info = self._success_info(obs)
        info["seed"] = seed
        info["hold_action"] = self.action_adapter.current_active_hold_action()
        return make_success_response(
            request,
            {
                "observation": formatted,
                "info": info,
                "episode_id": episode_id,
                "step_index": 0,
            },
        )

    def _handle_step(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        self.episode.validate_step(str(payload.get("episode_id")), int(payload.get("step_index", -1)))
        sim_action, action_info = self.action_adapter.to_sim_action(payload.get("action"))
        obs, reward, done, info = self.env.step(sim_action, verbose=False)
        return self._transition_response(request, obs, reward, done, info or {}, action_info)

    def _handle_step_native_action(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enable_debug_commands:
            raise RoCoProtocolError("Debug command is disabled.", code=ErrorCode.DEBUG_COMMAND_DISABLED)
        self.episode.validate_step(str(payload.get("episode_id")), int(payload.get("step_index", -1)))
        model = self.env.physics.model
        model_neq = int(getattr(model, "neq", len(getattr(model, "eq_active", []))))
        sim_action = decode_sim_action(payload.get("sim_action", {}), model_nu=int(model.nu), model_neq=model_neq)
        obs, reward, done, info = self.env.step(sim_action, verbose=False)
        return self._transition_response(request, obs, reward, done, info or {}, {"native_action": True})

    def _transition_response(
        self,
        request: Dict[str, Any],
        obs: Any,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        action_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        next_step_index = self.episode.step_index + 1
        truncated = next_step_index >= self.config.max_episode_steps and not bool(done)
        success_info = self._success_info(obs, reward=reward, done=done)
        info = dict(info)
        info.update(success_info)
        info.update(action_info)
        terminated = bool(done or info["is_success"])
        self.episode.advance_step(terminated_or_truncated=bool(terminated or truncated))
        formatted = self.observation_adapter.format(obs)
        self._show_live_observation(formatted)
        return make_success_response(
            request,
            {
                "observation": formatted,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": bool(truncated),
                "info": info,
                "episode_id": self.episode.episode_id,
                "step_index": self.episode.step_index,
            },
        )

    def _handle_render(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        image = self.observation_adapter.render()
        self._show_live_image(image)
        return make_success_response(request, {"image": image})

    def _handle_get_state_digest(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self.env.physics.data
        qpos = np.ascontiguousarray(data.qpos, dtype=np.float64)
        qvel = np.ascontiguousarray(data.qvel, dtype=np.float64)
        ctrl = np.ascontiguousarray(data.ctrl, dtype=np.float64)
        return make_success_response(
            request,
            {
                "qpos_sha256": _digest_array(qpos),
                "qvel_sha256": _digest_array(qvel),
                "ctrl_sha256": _digest_array(ctrl),
                "qpos": qpos,
                "qvel": qvel,
                "ctrl": ctrl,
                "timestep": int(getattr(self.env, "timestep", 0)),
            },
        )

    def _handle_close_episode(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        self.episode.close_episode()
        return make_success_response(request, {"status": "ready"})

    def _handle_shutdown(self, request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.allow_remote_shutdown:
            raise RoCoEpisodeError("Remote shutdown is disabled.", code=ErrorCode.UNAUTHORIZED_SHUTDOWN)
        if self.config.session_token is not None and payload.get("session_token") != self.config.session_token:
            raise RoCoEpisodeError("Invalid shutdown token.", code=ErrorCode.UNAUTHORIZED_SHUTDOWN)
        self._running = False
        self.episode.shutdown()
        return make_success_response(request, {"status": "closed"})

    def serve_forever(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required to run the RoCo bridge server.") from exc

        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, int(self.config.request_timeout_ms))
        socket.setsockopt(zmq.SNDTIMEO, int(self.config.request_timeout_ms))
        socket.bind(self.config.endpoint)
        self._running = True
        LOGGER.info("RoCo bridge server listening endpoint=%s task=%s active_agent=%s", self.config.endpoint, self.config.task, self.config.active_agent)
        try:
            while self._running:
                try:
                    data = socket.recv()
                except zmq.Again:
                    continue
                request: Optional[Dict[str, Any]] = None
                try:
                    request = unpack_message(data, max_payload_bytes=self.config.max_payload_bytes)
                    response = self.handle_request(request)
                except Exception as exc:
                    response = make_exception_response(request, exc)
                socket.send(pack_message(response, max_payload_bytes=self.config.max_payload_bytes))
        finally:
            if self._live_viewer is not None:
                self._live_viewer.close()
            socket.close(0)
