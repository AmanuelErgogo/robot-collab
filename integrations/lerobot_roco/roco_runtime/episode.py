"""Server-side episode state machine."""

import uuid
from dataclasses import dataclass
from typing import Optional

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoEpisodeError


class EpisodeStatus:
    STARTING = "STARTING"
    READY = "READY"
    EPISODE_ACTIVE = "EPISODE_ACTIVE"
    EPISODE_TERMINATED = "EPISODE_TERMINATED"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


@dataclass
class EpisodeStateMachine:
    state: str = EpisodeStatus.STARTING
    episode_id: Optional[str] = None
    step_index: int = 0

    def mark_ready(self) -> None:
        if self.state in {EpisodeStatus.STARTING, EpisodeStatus.EPISODE_ACTIVE, EpisodeStatus.EPISODE_TERMINATED}:
            self.state = EpisodeStatus.READY
            self.episode_id = None
            self.step_index = 0
            return
        if self.state == EpisodeStatus.READY:
            return
        raise RoCoEpisodeError("Cannot mark episode ready from current state.", details={"state": self.state})

    def reset(self) -> str:
        if self.state == EpisodeStatus.EPISODE_ACTIVE:
            raise RoCoEpisodeError(
                "Episode is already active.",
                code=ErrorCode.EPISODE_ALREADY_ACTIVE,
                details={"episode_id": self.episode_id, "state": self.state},
            )
        if self.state in {EpisodeStatus.CLOSED, EpisodeStatus.ERROR}:
            raise RoCoEpisodeError("Cannot reset from current state.", details={"state": self.state})
        self.state = EpisodeStatus.EPISODE_ACTIVE
        self.episode_id = str(uuid.uuid4())
        self.step_index = 0
        return self.episode_id

    def validate_step(self, episode_id: str, step_index: int) -> None:
        if self.state != EpisodeStatus.EPISODE_ACTIVE:
            code = ErrorCode.EPISODE_TERMINATED if self.state == EpisodeStatus.EPISODE_TERMINATED else ErrorCode.EPISODE_NOT_ACTIVE
            raise RoCoEpisodeError(
                "Episode is not active.",
                code=code,
                details={"state": self.state, "episode_id": self.episode_id},
            )
        if episode_id != self.episode_id:
            raise RoCoEpisodeError(
                "Episode ID mismatch.",
                code=ErrorCode.EPISODE_ID_MISMATCH,
                details={"expected": self.episode_id, "received": episode_id},
            )
        if int(step_index) != self.step_index:
            raise RoCoEpisodeError(
                "Step index mismatch.",
                code=ErrorCode.STEP_INDEX_MISMATCH,
                details={"expected": self.step_index, "received": step_index},
            )

    def advance_step(self, terminated_or_truncated: bool = False) -> int:
        self.step_index += 1
        if terminated_or_truncated:
            self.state = EpisodeStatus.EPISODE_TERMINATED
        return self.step_index

    def close_episode(self) -> None:
        if self.state in {EpisodeStatus.EPISODE_ACTIVE, EpisodeStatus.EPISODE_TERMINATED, EpisodeStatus.READY}:
            self.state = EpisodeStatus.READY
            self.episode_id = None
            self.step_index = 0
            return
        if self.state == EpisodeStatus.CLOSED:
            return
        raise RoCoEpisodeError("Cannot close episode from current state.", details={"state": self.state})

    def shutdown(self) -> None:
        self.state = EpisodeStatus.CLOSED
        self.episode_id = None
        self.step_index = 0
