# Interface and API Specification

The purpose of these interfaces is to make simulation, human-in-simulation, and real-world execution swappable without changing the controller logic.

## 1. Data types

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

TeamType = Literal["RR", "HR"]
CoordinationMode = Literal["Cent", "Dialog"]
SkillBackend = Literal["RRT", "LearnedSkill"]
MonitorBackend = Literal["VLM-self", "CodedSim", "SARM"]
EnvironmentType = Literal["sim", "real"]

class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    STAGE_DONE = "stage_done"
    TASK_DONE = "task_done"
    STUCK = "stuck"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ObservationBundle:
    rgb: dict[str, Any]
    depth: dict[str, Any] | None = None
    proprioception: dict[str, Any] = field(default_factory=dict)
    public_percepts: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    # Strictly monitor/evaluation only. Never pass this to non-oracle planner prompts.
    oracle_state: dict[str, Any] | None = None

@dataclass
class SkillCall:
    agent_id: str
    skill_name: str
    args: dict[str, Any]
    stage_id: str | None = None
    timeout_s: float | None = None

@dataclass
class ExecutionFeedback:
    agent_id: str
    skill_call: SkillCall | None
    status: Literal["running", "success", "failure", "timeout"]
    done: bool | None = None
    message: str = ""
    raw_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitorDecision:
    stage_id: str
    status: StageStatus
    progress_score: float
    should_replan: bool = False
    should_retry: bool = False
    is_stage_done: bool = False
    is_task_done: bool = False
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    monitor_backend: MonitorBackend = "CodedSim"
    privileged: bool = False

@dataclass
class HumanInstruction:
    text: str
    target_human_id: str = "human_1"
    expected_stage_id: str | None = None

@dataclass
class HumanResponse:
    text: str
    action: Literal["accept", "reject", "counter_propose", "done", "unknown"] = "unknown"
    timestamp: float = 0.0

@dataclass
class PlanStep:
    step_id: str
    agent_id: str
    skill_call: SkillCall | None = None
    human_instruction: HumanInstruction | None = None
    preconditions: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)

@dataclass
class PlanUpdate:
    steps: list[PlanStep]
    dialogue_message: str | None = None
    rationale: str = ""
```

## 2. Environment adapter

```python
class EnvironmentAdapter:
    def reset(self, task_id: str, seed: int) -> ObservationBundle:
        raise NotImplementedError

    def observe(self) -> ObservationBundle:
        raise NotImplementedError

    def step_robot(self, agent_id: str, low_level_action: Any) -> ObservationBundle:
        raise NotImplementedError

    def apply_human_action(self, human_id: str, action: Any) -> ObservationBundle:
        raise NotImplementedError

    def get_task_done(self) -> bool | None:
        raise NotImplementedError

    def get_oracle_state(self) -> dict[str, Any]:
        raise NotImplementedError
```

Implementation notes:

- In simulation, `get_oracle_state()` can expose object poses, grasp state, target state, done, and task predicates.
- In real world, `get_oracle_state()` should be empty or limited to evaluation-only annotations.
- `ObservationBundle.oracle_state` should be stripped before planner prompts in non-oracle conditions.

## 3. Planner interface

```python
class PlannerInterface:
    def reset_episode(self, task_id: str, goal: str, capabilities: dict[str, Any]) -> None:
        pass

    def propose_next(
        self,
        observation: ObservationBundle,
        goal: str,
        history: list[dict[str, Any]],
        dialogue: list[dict[str, str]],
        feedback: ExecutionFeedback | None = None,
        monitor_decision: MonitorDecision | None = None,
    ) -> PlanUpdate:
        raise NotImplementedError
```

Baseline planners receive `monitor_decision=None`. They self-monitor from image and previous execution status.

CRIE-BT planners may receive monitor decisions when replanning or updating the BT.

## 4. Skill executor interface

```python
class SkillExecutorInterface:
    backend_name: str

    def reset(self, env: EnvironmentAdapter) -> None:
        pass

    def start(self, skill_call: SkillCall, observation: ObservationBundle) -> None:
        raise NotImplementedError

    def step(self, observation: ObservationBundle) -> ExecutionFeedback:
        raise NotImplementedError

    def stop(self) -> None:
        pass
```

Backends:

- `RRTSkillExecutor`: Step 1 and Step 2.
- `LearnedSkillExecutor`: Step 3; VLA/ACT/Octo/LeRobot-like skill policy.

## 5. Progress monitor interface

```python
class ProgressMonitorInterface:
    backend_name: str
    privileged: bool

    def reset_stage(self, stage_id: str, skill_call: SkillCall | None, observation: ObservationBundle) -> None:
        pass

    def update(
        self,
        observation: ObservationBundle,
        feedback: ExecutionFeedback | None,
        history: list[dict[str, Any]],
    ) -> MonitorDecision:
        raise NotImplementedError
```

Backends:

```python
class CodedSimProgressMonitor(ProgressMonitorInterface):
    backend_name = "CodedSim"
    privileged = True

class SARMProgressMonitor(ProgressMonitorInterface):
    backend_name = "SARM"
    privileged = False

class NoSeparateMonitor(ProgressMonitorInterface):
    backend_name = "VLM-self"
    privileged = False
```

`NoSeparateMonitor` should normally not be used by the baseline controller except as a logging placeholder.

## 6. Communication interface

```python
class CommunicationInterface:
    def send_to_human(self, instruction: HumanInstruction) -> None:
        raise NotImplementedError

    def read_human_response(self, timeout_s: float | None = None) -> HumanResponse:
        raise NotImplementedError

    def send_robot_dialogue(self, speaker_id: str, text: str) -> None:
        raise NotImplementedError
```

Initial implementation:

- `TerminalCommunicationInterface` using printed messages and `input()` or nonblocking key commands.

Later wrappers:

- `SpeechCommunicationInterface` using TTS/STT.
- `GUICommunicationInterface` if needed.

## 7. Controller interface

```python
class CollaborationController:
    def run_episode(
        self,
        condition_name: str,
        env: EnvironmentAdapter,
        task_id: str,
        goal: str,
        max_steps: int,
    ) -> dict[str, Any]:
        raise NotImplementedError
```

Concrete controllers:

- `VLMCentralizedController`
- `VLMDialogController`
- `CRIEBTCentralizedController`
- `CRIEBTDialogController`

## 8. Fairness enforcement

Add a gate before planner calls:

```python
def planner_safe_observation(obs: ObservationBundle) -> ObservationBundle:
    return ObservationBundle(
        rgb=obs.rgb,
        depth=obs.depth,
        proprioception=obs.proprioception,
        public_percepts=obs.public_percepts,
        timestamp=obs.timestamp,
        oracle_state=None,
    )
```

Only the coded simulator monitor and evaluator should receive `oracle_state`.
