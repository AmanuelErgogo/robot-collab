"""Central joint action merge and single-step execution."""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from .models import AgentActionFragment, JointStepResult


class JointActionMergeError(RuntimeError):
    pass


class CentralJointStepper(object):
    def __init__(
        self,
        env: Any,
        agent_order: Sequence[str],
        sim_action_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.env = env
        self.agent_order = tuple(agent_order)
        self.step_index = 0
        self._sim_action_factory = sim_action_factory

    def merge(self, fragments: Mapping[str, AgentActionFragment]) -> Any:
        ctrl_by_index = {}
        qpos_by_index = {}
        eq_by_index = {}
        ordered_agents = [agent for agent in self.agent_order if agent in fragments]
        ordered_agents.extend(sorted(agent for agent in fragments if agent not in ordered_agents))

        for agent in ordered_agents:
            arrays = fragments[agent].arrays()
            self._merge_values(ctrl_by_index, arrays["ctrl_idxs"], arrays["ctrl_vals"], "ctrl", agent)
            self._merge_values(qpos_by_index, arrays["qpos_idxs"], arrays["qpos_target"], "qpos", agent)
            self._merge_values(eq_by_index, arrays["eq_active_idxs"], arrays["eq_active_vals"], "eq_active", agent)

        ctrl_idxs = np.ascontiguousarray(sorted(ctrl_by_index), dtype=np.int32)
        ctrl_vals = np.ascontiguousarray([ctrl_by_index[int(idx)] for idx in ctrl_idxs], dtype=np.float32)
        qpos_idxs = np.ascontiguousarray(sorted(qpos_by_index), dtype=np.int32)
        qpos_target = np.ascontiguousarray([qpos_by_index[int(idx)] for idx in qpos_idxs], dtype=np.float32)
        eq_active_idxs = np.ascontiguousarray(sorted(eq_by_index), dtype=np.int32)
        eq_active_vals = np.ascontiguousarray([eq_by_index[int(idx)] for idx in eq_active_idxs], dtype=np.int32)
        sim_action_factory = self._sim_action_factory or self._default_sim_action_factory()
        return sim_action_factory(
            ctrl_idxs=ctrl_idxs,
            ctrl_vals=ctrl_vals,
            qpos_idxs=qpos_idxs,
            qpos_target=qpos_target,
            eq_active_idxs=eq_active_idxs if len(eq_active_idxs) else None,
            eq_active_vals=eq_active_vals if len(eq_active_vals) else None,
        )

    def step(self, fragments: Mapping[str, AgentActionFragment]) -> JointStepResult:
        sim_action = self.merge(fragments)
        obs, reward, done, info = self.env.step(sim_action, verbose=False)
        self.step_index += 1
        return JointStepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
            step_index=self.step_index,
            sim_action=sim_action,
            merged_agent_order=tuple(agent for agent in self.agent_order if agent in fragments),
        )

    def _merge_values(self, dest: Dict[int, Any], idxs: Any, vals: Any, name: str, agent: str) -> None:
        idx_arr = np.asarray(idxs, dtype=np.int32)
        val_arr = np.asarray(vals)
        if len(idx_arr) != len(val_arr):
            raise JointActionMergeError("{} index/value length mismatch for {}.".format(name, agent))
        for idx, val in zip(idx_arr, val_arr):
            idx_int = int(idx)
            if idx_int in dest and not np.allclose(dest[idx_int], val):
                raise JointActionMergeError("{} index {} has conflicting values.".format(name, idx_int))
            dest[idx_int] = val

    def _default_sim_action_factory(self) -> Callable[..., Any]:
        from rocobench.envs.base_env import SimAction

        return SimAction
