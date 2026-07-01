"""MockACTHandle — an ACT-interface-compatible policy backed by the existing RRT planner.

This lets the full LearnedSkillExecutor + SubtaskLearnedExecutor pipeline run
end-to-end in simulation without a real neural-network checkpoint.

Swap this out for a real LeRobot ACT checkpoint by implementing the same
LearnedPolicyHandle interface (reset / predict_native_chunk).

Usage
-----
    handle = MockACTHandle(
        env=env,
        robots=robots,
        chunk_size=10,
        uncertainty_mode="policy_metadata",
    )
    cache = BoundedPolicyHandleCache(loader=lambda spec: handle, max_size=1)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from rocobench.envs.base_env import SimAction
from rocobench.policy import PlannedPathPolicy
from rocobench.subtask_plan import LLMPathPlan

from .errors import PolicyHandleError
from .policy_handle import LearnedPolicyHandle, NativeActionChunk


class MockACTHandle(LearnedPolicyHandle):
    """
    Deterministic mock of an ACT policy.

    On each call to predict_native_chunk, it:
      1. Parses the instruction to find the target EE pose.
      2. Runs the RRT planner to build a joint-space trajectory.
      3. Returns the next `chunk_size` steps as a [chunk_size × n_ctrl] numpy array.
      4. Optionally injects synthetic confidence metadata for uncertainty testing.

    The action encoding is a flat array of MuJoCo ctrl values. The
    SubtaskEnvAdapter knows how to convert this back into a SimAction.
    """

    def __init__(
        self,
        env,
        robots: Dict[str, Any],
        chunk_size: int = 10,
        uncertainty_mode: str = "policy_metadata",
        base_confidence: float = 0.92,
        noise_std: float = 0.0,
    ):
        self.env = env
        self.robots = robots
        self.chunk_size = int(chunk_size)
        self.uncertainty_mode = uncertainty_mode
        self.base_confidence = float(base_confidence)
        self.noise_std = float(noise_std)
        # Map from display agent name ("Alice") → hardware body name ("ur5e_robotiq")
        # Some envs expose robot_name_map_inv; fall back to identity if absent.
        self._agent_to_hw = getattr(env, "robot_name_map_inv", {})
        # obs attributes use hardware names (ur5e_robotiq, panda)
        self._hw_names = set(self._agent_to_hw.values()) if self._agent_to_hw else set(robots.keys())
        # Pre-collect ctrl indices in the same order the SubtaskEnvAdapter uses,
        # so _sim_action_to_array can produce correctly-aligned output.
        self._ctrl_idxs_order = np.array(
            [idx for robot in robots.values() for idx in robot.joint_idxs_in_ctrl],
            dtype=np.int32,
        )

        # State across predict calls for one episode
        self._action_buffer: list[SimAction] = []
        self._action_idx: int = 0
        self._current_instruction: Optional[str] = None
        self._plan_success: bool = False
        self._plan_time: float = 0.0

    # ------------------------------------------------------------------
    # LearnedPolicyHandle interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._action_buffer = []
        self._action_idx = 0
        self._current_instruction = None
        self._plan_success = False
        self._plan_time = 0.0

    def predict_native_chunk(
        self,
        observation,
        instruction: Mapping[str, Any],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> NativeActionChunk:
        # Replan only when the instruction changes; once planned, hold position on exhaustion
        canonical = instruction.get("canonical", "") if isinstance(instruction, dict) else str(instruction)
        if canonical != self._current_instruction:
            self._replan(observation, instruction)
            self._current_instruction = canonical

        # Slice the next chunk
        start = self._action_idx
        end = min(start + self.chunk_size, len(self._action_buffer))
        chunk_actions = self._action_buffer[start:end]
        self._action_idx = end

        eq_active_per_step = []
        if not chunk_actions:
            # Plan exhausted — hold last position so executor's no-progress monitor fires
            if self._action_buffer:
                last = self._sim_action_to_array(self._action_buffer[-1], self._ctrl_idxs_order)
                actions_arr = np.tile(last, (self.chunk_size, 1)).astype(np.float32)
            else:
                # Nothing planned at all — clamp zeros to action bounds
                n_ctrl = len(action_low)
                actions_arr = np.clip(
                    np.zeros((self.chunk_size, n_ctrl), dtype=np.float32),
                    action_low, action_high,
                )
            eq_active_per_step = [None] * self.chunk_size
        else:
            flat_actions = [self._sim_action_to_array(a, self._ctrl_idxs_order) for a in chunk_actions]
            # Collect per-step eq_active (weld/gripper commands)
            for sa in chunk_actions:
                idxs = getattr(sa, "eq_active_idxs", None)
                vals = getattr(sa, "eq_active_vals", None)
                if idxs is not None and len(idxs) > 0:
                    eq_active_per_step.append((idxs.tolist(), vals.tolist()))
                else:
                    eq_active_per_step.append(None)
            actions_arr = np.stack(flat_actions, axis=0).astype(np.float32)
            # Pad last partial chunk to full chunk_size
            if len(actions_arr) < self.chunk_size:
                pad = np.repeat(actions_arr[-1:], self.chunk_size - len(actions_arr), axis=0)
                actions_arr = np.concatenate([actions_arr, pad], axis=0)
                eq_active_per_step += [None] * (self.chunk_size - len(chunk_actions))

        if self.noise_std > 0:
            actions_arr += np.random.randn(*actions_arr.shape).astype(np.float32) * self.noise_std

        # Clip to bounds to avoid validation failure from float32 rounding
        actions_arr = np.clip(actions_arr, action_low, action_high)

        metadata = self._build_metadata()
        metadata["eq_active_per_step"] = eq_active_per_step
        return NativeActionChunk(actions=actions_arr, metadata=metadata)

    def health_check(self, spec) -> Dict[str, Any]:
        policy_type = getattr(spec, "policy_type", None) if spec is not None else None
        if not isinstance(policy_type, str):
            policy_type = "act"
        return {
            "policy_type": policy_type,
            "chunk_size": self.chunk_size,
            "schema_hash": spec.schema_hash if spec is not None else "mock",
            "action_representation": "joint_ctrl",
            "backend": "mock_rrt",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replan(self, observation, instruction: Any) -> None:
        """Build a new RRT-planned action buffer for the current instruction."""
        self._action_buffer = []
        self._action_idx = 0

        skill_name, call_args = self._parse_instruction(instruction)
        path_plan = self._build_path_plan(observation, skill_name, call_args)
        if path_plan is None:
            self._plan_success = False
            return

        t0 = time.monotonic()
        try:
            policy = PlannedPathPolicy(
                physics=self.env.physics,
                robots=self.robots,
                path_plan=path_plan,
                graspable_object_names=self.env.get_graspable_objects(),
                allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                timeout=8,   # cap RRT search at 8 seconds
                control_freq=1,  # keep all path nodes for smooth simulation tracking
            )
            ok, _ = policy.plan(self.env)
            self._plan_success = ok
            if ok:
                self._action_buffer = list(policy.action_buffer)
        except Exception as exc:
            print(f"[MockACTHandle] RRT planning failed: {exc}")
            self._plan_success = False
        finally:
            self._plan_time = time.monotonic() - t0

    def _parse_instruction(self, instruction: Any):
        """Extract skill_name and call_args from instruction dict or string."""
        if isinstance(instruction, dict):
            canonical = instruction.get("canonical", "")
        else:
            canonical = str(instruction)

        # canonical looks like "PICK(object=apple)" or "PLACE(target=cutting_board)"
        if "(" in canonical:
            skill_name = canonical.split("(")[0].strip().upper()
            args_str = canonical.split("(", 1)[1].rstrip(")")
            call_args = {}
            for part in args_str.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    call_args[k.strip()] = v.strip()
        else:
            skill_name = canonical.upper()
            call_args = {}
        return skill_name, call_args

    def _build_path_plan(self, obs, skill_name: str, call_args: Dict[str, str]):
        """
        Translate (skill_name, call_args) → LLMPathPlan.

        This mimics what the parser does but is driven directly by the env's
        geometric helpers rather than LLM text parsing.
        """
        agent_names = list(self.robots.keys())
        if not agent_names:
            return None

        # Primary agent is the first non-waiting robot. We pick the first.
        primary_agent = agent_names[0]
        # obs attributes use hardware names; resolve if needed
        primary_hw = self._agent_to_hw.get(primary_agent, primary_agent)

        if skill_name in ("PICK", "PICK_AND_PLACE"):
            return self._pick_plan(obs, primary_agent, primary_hw, call_args)
        if skill_name == "PLACE":
            return self._place_plan(obs, primary_agent, primary_hw, call_args)
        if skill_name == "STACK_ON":
            return self._stack_plan(obs, primary_agent, primary_hw, call_args)
        if skill_name == "OPEN_CABINET":
            return self._open_plan(obs, primary_agent, primary_hw, call_args)
        # WAIT or unknown — no motion
        return None

    def _make_plan(self, obs, agent_name: str, target_pos, target_quat, obj_name=None,
                   obj_site=None, grasp_val=1, inhand=None, hw_name: str = None):
        """Build a minimal LLMPathPlan for a single EE target."""
        agent_names = list(self.robots.keys())
        # obs uses hardware body names, not display agent names
        obs_key = hw_name or self._agent_to_hw.get(agent_name, agent_name)
        agent_state = getattr(obs, obs_key, None)
        if agent_state is None:
            return None

        current_ee = np.array(agent_state.ee_pose)
        target_pose = np.concatenate([target_pos, target_quat])

        # 3-waypoint path: hover → pre-grasp (2cm above) → target
        hover_pos = target_pos.copy()
        hover_pos[2] = max(target_pos[2] + 0.10, 0.3)
        hover_pose = np.concatenate([hover_pos, target_quat])

        pre_grasp_pos = target_pos.copy()
        pre_grasp_pos[2] = target_pos[2] + 0.015  # just above, so weld fires
        pre_grasp_pose = np.concatenate([pre_grasp_pos, target_quat])

        ee_targets = {agent_name: target_pose}
        ee_waypoints = {agent_name: [hover_pose, pre_grasp_pose]}
        tograsp = {agent_name: None}
        inhand_d = {agent_name: None}

        if obj_name is not None and obj_site is not None:
            tograsp[agent_name] = (obj_name, obj_site, grasp_val)
        if inhand is not None:
            inhand_d[agent_name] = inhand

        # How many waypoints does the primary agent have?
        n_waypoints = len(ee_waypoints[agent_name])

        # All other agents stay in place (WAIT) — must match waypoint count
        for other in agent_names:
            if other == agent_name:
                continue
            other_hw = self._agent_to_hw.get(other, other)
            other_state = getattr(obs, other_hw, None)
            if other_state is None:
                continue
            other_ee = np.array(other_state.ee_pose)
            ee_targets[other] = other_ee
            # Repeat current pose for each waypoint slot so lengths match
            ee_waypoints[other] = [other_ee] * n_waypoints
            tograsp[other] = None
            inhand_d[other] = None

        return LLMPathPlan(
            agent_names=agent_names,
            ee_targets=ee_targets,
            ee_waypoints=ee_waypoints,
            tograsp=tograsp,
            inhand=inhand_d,
            parsed_proposal=f"mock:{self._current_instruction}",
            action_strs={a: "" for a in agent_names},
            return_home=None,
        )

    def _pick_plan(self, obs, agent_name, hw_name, call_args):
        obj_name = call_args.get("object", "")
        obj_state = obs.objects.get(obj_name)
        if obj_state is None:
            return None
        site_name = self.env.get_grasp_site(obj_name)
        if site_name is None:
            return None
        target_pos = obj_state.sites[site_name].xpos.copy()
        target_quat = obj_state.sites[site_name].xquat.copy()
        return self._make_plan(obs, agent_name, target_pos, target_quat,
                               obj_name=obj_name, obj_site=site_name, grasp_val=1,
                               hw_name=hw_name)

    def _place_plan(self, obs, agent_name, hw_name, call_args):
        target_name = call_args.get("target", "")
        agent_state = getattr(obs, hw_name, None)
        if agent_state is None:
            return None
        target_pos = self.env.get_target_pos(agent_name, target_name)
        if target_pos is None:
            return None
        target_quat = self.env.get_target_quat(agent_name, target_name)
        inhand = self._get_inhand(obs, hw_name)
        obj_site = self.env.get_grasp_site(inhand[0]) if inhand else None
        return self._make_plan(obs, agent_name, np.array(target_pos), target_quat,
                               obj_name=inhand[0] if inhand else None,
                               obj_site=obj_site, grasp_val=0,
                               inhand=inhand, hw_name=hw_name)

    def _stack_plan(self, obs, agent_name, hw_name, call_args):
        target_name = call_args.get("target", "")
        target_state = obs.objects.get(target_name)
        if target_state is None:
            return None
        target_pos = target_state.xpos.copy()
        target_pos[2] = target_state.top_height + 0.02
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        inhand = self._get_inhand(obs, hw_name)
        obj_site = self.env.get_grasp_site(inhand[0]) if inhand else None
        return self._make_plan(obs, agent_name, target_pos, target_quat,
                               obj_name=inhand[0] if inhand else None,
                               obj_site=obj_site, grasp_val=0,
                               inhand=inhand, hw_name=hw_name)

    def _open_plan(self, obs, agent_name, hw_name, call_args):
        door = call_args.get("door", "")
        if not door.endswith("_handle"):
            door += "_handle"
        target_pos = getattr(self.env, "open_pose", {}).get(door)
        if target_pos is None:
            return None
        target_pos = np.array(target_pos[:3])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        return self._make_plan(obs, agent_name, target_pos, target_quat, hw_name=hw_name)

    def _get_inhand(self, obs, hw_name):
        """Return (obj_name, site_name, joint_name) if the agent is holding something."""
        agent_state = getattr(obs, hw_name, None)
        if agent_state is None:
            return None
        contacts = agent_state.contacts
        for obj_name in (self.env.get_graspable_objects() or []):
            if isinstance(obj_name, dict):
                items = list(obj_name.values())
                for sub in items:
                    for o in (sub if isinstance(sub, list) else [sub]):
                        if o in contacts:
                            site = self.env.get_grasp_site(o)
                            return (o, site, None) if site else None
            else:
                if obj_name in contacts:
                    site = self.env.get_grasp_site(obj_name)
                    return (obj_name, site, None) if site else None
        return None

    def _sim_action_to_array(
        self, sim_action: SimAction, ctrl_idxs_order: np.ndarray
    ) -> np.ndarray:
        """Convert a SimAction to a flat array aligned with ctrl_idxs_order.

        The output array has one entry per element of ctrl_idxs_order, ordered
        the same way the SubtaskEnvAdapter.action_bounds returns bounds.
        """
        n = len(ctrl_idxs_order)
        arr = np.zeros(n, dtype=np.float32)
        # Build a lookup: absolute ctrl_idx → value from this SimAction
        action_map = {int(i): float(v)
                      for i, v in zip(sim_action.ctrl_idxs, sim_action.ctrl_vals)}
        for pos, ctrl_idx in enumerate(ctrl_idxs_order):
            if int(ctrl_idx) in action_map:
                arr[pos] = action_map[int(ctrl_idx)]
        return arr

    def _build_metadata(self) -> Dict[str, Any]:
        """Synthetic policy metadata for uncertainty estimation."""
        if self.uncertainty_mode == "policy_metadata":
            # Simulate CVAE KL-divergence spread: confident if plan succeeded
            confidence = self.base_confidence if self._plan_success else 0.35
            # Add realistic jitter
            confidence = float(np.clip(
                confidence + np.random.randn() * 0.03, 0.05, 0.99
            ))
            return {
                "confidence": confidence,
                "plan_time_s": self._plan_time,
                "plan_success": self._plan_success,
                "backend": "mock_rrt",
            }
        return {
            "plan_time_s": self._plan_time,
            "plan_success": self._plan_success,
            "backend": "mock_rrt",
        }
