from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import dataclasses, validator

from rocobench.envs.env_utils import Pose


class AllowArbitraryTypes:
    arbitrary_types_allowed = True


class BehaviorStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclasses.dataclass(frozen=False)
class BehaviorContext:
    blackboard: Dict[str, Any]


class BehaviorNode(ABC):
    def __init__(self, name: str):
        self.name = name
        self.status = BehaviorStatus.IDLE

    @abstractmethod
    def tick(self, context: BehaviorContext) -> BehaviorStatus:
        raise NotImplementedError

    def reset(self):
        self.status = BehaviorStatus.IDLE


class SequenceNode(BehaviorNode):
    def __init__(self, name: str, children: Sequence[BehaviorNode]):
        super().__init__(name=name)
        self.children = list(children)
        self.current_child_idx = 0

    def tick(self, context: BehaviorContext) -> BehaviorStatus:
        while self.current_child_idx < len(self.children):
            child = self.children[self.current_child_idx]
            child_status = child.tick(context)
            if child_status == BehaviorStatus.SUCCESS:
                self.current_child_idx += 1
                continue
            self.status = child_status
            return self.status

        self.status = BehaviorStatus.SUCCESS
        return self.status

    def reset(self):
        super().reset()
        self.current_child_idx = 0
        for child in self.children:
            child.reset()


class FallbackNode(BehaviorNode):
    def __init__(self, name: str, children: Sequence[BehaviorNode]):
        super().__init__(name=name)
        self.children = list(children)
        self.current_child_idx = 0

    def tick(self, context: BehaviorContext) -> BehaviorStatus:
        while self.current_child_idx < len(self.children):
            child = self.children[self.current_child_idx]
            child_status = child.tick(context)
            if child_status == BehaviorStatus.FAILURE:
                self.current_child_idx += 1
                continue
            self.status = child_status
            return self.status

        self.status = BehaviorStatus.FAILURE
        return self.status

    def reset(self):
        super().reset()
        self.current_child_idx = 0
        for child in self.children:
            child.reset()


class ConditionNode(BehaviorNode):
    def __init__(self, name: str, predicate: Callable[[BehaviorContext], bool]):
        super().__init__(name=name)
        self.predicate = predicate

    def tick(self, context: BehaviorContext) -> BehaviorStatus:
        self.status = (
            BehaviorStatus.SUCCESS if self.predicate(context) else BehaviorStatus.FAILURE
        )
        return self.status


class RetryNode(BehaviorNode):
    def __init__(self, name: str, child: BehaviorNode, max_attempts: int):
        super().__init__(name=name)
        self.child = child
        self.max_attempts = max_attempts
        self.attempts = 0

    def tick(self, context: BehaviorContext) -> BehaviorStatus:
        child_status = self.child.tick(context)
        if child_status == BehaviorStatus.FAILURE:
            self.attempts += 1
            if self.attempts >= self.max_attempts:
                self.status = BehaviorStatus.FAILURE
                return self.status
            self.child.reset()
            self.status = BehaviorStatus.RUNNING
            return self.status
        self.status = child_status
        return self.status

    def reset(self):
        super().reset()
        self.attempts = 0
        self.child.reset()


@dataclasses.dataclass(frozen=False)
class MotionPrimitive:
    agent_names: List[str]
    ee_waypoints: Dict[str, List]
    tograsp: Dict[str, Union[Tuple, None]]
    inhand: Dict[str, Union[Tuple, None]]
    ee_targets: Dict
    parsed_proposal: str
    action_strs: Dict[str, str]
    return_home: Optional[Dict[str, bool]]

    @validator("ee_waypoints")
    def check_waypoints_shape(cls, v):
        for robot_name, waypoints in v.items():
            for waypoint in waypoints:
                assert len(waypoint) == 7, "waypoint should be a 3-dim pos, 4-dim quat"
        return v

    @validator("ee_targets")
    def check_ee_targets_shape(cls, v):
        for robot_name, ee_pose in v.items():
            assert len(ee_pose) == 7, "ee_pose should be a 3-dim pos, 4-dim quat"
        return v

    @validator("ee_waypoints")
    def check_same_length(cls, v):
        assert (
            len(set([len(waypoints) for waypoints in v.values()])) == 1
        ), "all robot waypoints should have same length"
        return v

    def get_robot_action_str(self, name) -> str:
        return self.action_strs.get(name, "")

    def get_action_desp(self):
        return "\n".join(
            f"{robot_name}: {self.action_strs[robot_name]}" for robot_name in self.agent_names
        )

    @property
    def num_ee_waypoints(self):
        if len(self.ee_waypoints) == 0:
            return 0
        return len(self.ee_waypoints[self.agent_names[0]])

    def convert_waypoints_dict_to_list(self) -> List[Dict[str, Pose]]:
        result = []
        for t in range(self.num_ee_waypoints):
            pose_dict = dict()
            for robot_name in self.agent_names:
                pose_arr = self.ee_waypoints[robot_name][t]
                pose_dict[robot_name] = Pose(
                    position=pose_arr[:3], orientation=pose_arr[3:]
                )
            result.append(pose_dict)
        return result

    def __post_init__(self):
        poses = dict()
        for name, target_arr in self.ee_targets.items():
            poses[name] = Pose(position=target_arr[:3], orientation=target_arr[3:])
        self.ee_target_poses = poses

        self.ee_waypoint_poses = dict()
        for name, waypoints in self.ee_waypoints.items():
            self.ee_waypoint_poses[name] = [
                Pose(position=waypoint[:3], orientation=waypoint[3:])
                for waypoint in waypoints
            ]

        self.ee_waypoints_list = self.convert_waypoints_dict_to_list()

        self.path_3d_list = []
        for name in self.agent_names:
            waypts_3d = [pose.position for pose in self.ee_waypoint_poses[name]]
            waypts_3d = waypts_3d + [self.ee_targets[name][:3]]
            self.path_3d_list.append(waypts_3d)

        if self.return_home is None:
            self.return_home = {name: False for name in self.agent_names}

    def get_inhand_ids(self, physics) -> Dict[str, List[int]]:
        all_bodies = []
        for i in range(physics.model.nbody):
            all_bodies.append(physics.model.body(i))
        inhand = dict()
        for robot_name, info in self.inhand.items():
            inhand[robot_name] = []
            if info is not None:
                obj_body_name = info[0]
                root_id = physics.model.body(obj_body_name).id
                obj_ids = [root_id]
                obj_ids += [
                    body.id for body in all_bodies if body.rootid[0] == root_id
                ]
                inhand[robot_name] = obj_ids

        return inhand

    def get_allowed_collision_ids(self, physics) -> Dict[str, List[int]]:
        inhand = self.get_inhand_ids(physics)

        all_bodies = []
        for i in range(physics.model.nbody):
            all_bodies.append(physics.model.body(i))
        for robot_name, info in self.tograsp.items():
            if info is not None:
                body_name = info[0]
                body_ids = [body.id for body in all_bodies if body.name == body_name]
                inhand[robot_name] += body_ids
        return inhand

    def get_inhand_obj_info(self, physics) -> Dict[str, Tuple]:
        inhand_info = dict()
        for robot_name, _info in self.inhand.items():
            if _info is not None:
                assert len(_info) == 3, (
                    "inhand info should be a tuple of "
                    "(obj_body_name, obj_site_name, obj_joint_name)"
                )
            inhand_info[robot_name] = _info
        return inhand_info


class MotionPrimitiveNode(BehaviorNode):
    def __init__(self, name: str, primitive: MotionPrimitive):
        super().__init__(name=name)
        self.primitive = primitive
        self._executor_key = f"executor::{self.name}"

    def tick(self, context: BehaviorContext) -> BehaviorStatus:
        blackboard = context.blackboard
        env = blackboard["env"]
        executor = blackboard.get(self._executor_key)

        if executor is None:
            validator = blackboard.get("primitive_validator")
            if validator is not None:
                valid, reason = validator(self.primitive)
                if not valid:
                    blackboard["last_failure"] = reason
                    self.status = BehaviorStatus.FAILURE
                    return self.status

            executor_factory = blackboard["executor_factory"]
            executor = executor_factory(self.primitive)
            plan_success, reason = executor.plan(env)
            if not plan_success:
                blackboard["last_failure"] = reason
                self.status = BehaviorStatus.FAILURE
                return self.status

            blackboard[self._executor_key] = executor
            blackboard["active_executor"] = executor
            blackboard["active_executor_key"] = self._executor_key
            blackboard.setdefault("planned_leaf_actions", []).append(
                list(executor.action_buffer)
            )
            blackboard.setdefault("planned_leaf_rrt", []).append(executor.rrt_plan_results)
            self.status = BehaviorStatus.RUNNING
            return self.status

        if executor.plan_exhausted:
            blackboard.pop(self._executor_key, None)
            if blackboard.get("active_executor_key") == self._executor_key:
                blackboard.pop("active_executor_key", None)
                blackboard.pop("active_executor", None)
            self.status = BehaviorStatus.SUCCESS
            return self.status

        blackboard["active_executor"] = executor
        blackboard["active_executor_key"] = self._executor_key
        self.status = BehaviorStatus.RUNNING
        return self.status

    def reset(self):
        super().reset()


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=False)
class BehaviorTreePlan:
    root: BehaviorNode
    motion_primitives: List[MotionPrimitive]
    agent_names: List[str]
    parsed_proposal: str
    action_strs: Dict[str, str]

    @classmethod
    def from_primitives(
        cls,
        motion_primitives: Sequence[MotionPrimitive],
        name: str = "llm_round",
    ) -> "BehaviorTreePlan":
        if len(motion_primitives) == 0:
            raise ValueError("BehaviorTreePlan requires at least one motion primitive.")
        root = SequenceNode(
            name=name,
            children=[
                MotionPrimitiveNode(name=f"{name}_primitive_{idx}", primitive=primitive)
                for idx, primitive in enumerate(motion_primitives)
            ],
        )
        head = motion_primitives[0]
        return cls(
            root=root,
            motion_primitives=list(motion_primitives),
            agent_names=head.agent_names,
            parsed_proposal=head.parsed_proposal,
            action_strs=head.action_strs,
        )

    def reset(self):
        self.root.reset()

    def iter_motion_primitives(self) -> Iterable[MotionPrimitive]:
        return iter(self.motion_primitives)

    @property
    def num_motion_primitives(self) -> int:
        return len(self.motion_primitives)

    def get_action_desp(self) -> str:
        return "\n".join(
            f"{robot_name}: {self.action_strs[robot_name]}" for robot_name in self.agent_names
        )
