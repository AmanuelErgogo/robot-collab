# Architecture Specification

## A. Baseline: monolithic VLM self-monitoring

The baseline planner performs task decomposition, allocation, next-action selection, progress assessment, and recovery in one VLM loop.

```text
Goal + agent capabilities + image observation + history + previous execution status
        ↓
VLM planner / self-monitor
        ↓
next robot skill call and/or human instruction
        ↓
RRT executor or human collaborator
        ↓
new image/status feedback
        ↺
```

### Baseline centralized

One VLM planner controls the whole team.

- `VLM-RR-Cent`: one planner assigns subtasks to Robot 1 and Robot 2.
- `VLM-HR-Cent`: one planner assigns subtasks to Robot and Human.

The centralized planner may produce:

- `robot_skill_call(agent="Robot1", skill="pick_place", args=...)`;
- `human_instruction("Please pick the red medication bottle and place it in the patient box.")`.

### Baseline dialog/distributed

Each robot-side agent has its own planner. Coordination occurs through dialogue.

- `VLM-RR-Dialog`: Robot 1 and Robot 2 each have a VLM planner and exchange messages.
- `VLM-HR-Dialog`: robot has a VLM planner and coordinates with a human through dialogue. The human is independent.

### Baseline input fairness rule

The baseline can use:

- image or rendered observation;
- capabilities;
- goal;
- prior actions;
- dialogue history;
- previous execution status, including success/failure/timeout/done.

The baseline cannot use coded simulator predicates, object poses, or oracle progress scores unless the run is explicitly labeled as an oracle ablation.

---

## B. Ours: CRIE-BT with explicit progress monitor

CRIE-BT separates high-level reasoning, structured execution, and progress monitoring.

```text
Goal + agent capabilities + image observation + history
        ↓
CRIE planner
        ↓
Behavior Tree
        ↓
Skill executor or human instruction
        ↓
Explicit progress monitor
        ↓
continue / retry / replan / ask human / update BT
        ↺
```

### CRIE-BT centralized

One central CRIE-BT controller manages a shared BT and assigns subtasks to both collaborators.

- `CRIE-BT-RR-Cent`: one central BT controls Robot 1 and Robot 2.
- `CRIE-BT-HR-Cent`: one central BT assigns robot skills and human instructions.

### CRIE-BT dialog/distributed

Each robot-side agent has its own CRIE-BT controller. Coordination happens through dialogue.

- `CRIE-BT-RR-Dialog`: Robot 1 and Robot 2 each run CRIE-BT and communicate.
- `CRIE-BT-HR-Dialog`: robot runs CRIE-BT and communicates with a human collaborator.

### Progress monitor backend by stage

| Stage | CRIE-BT monitor backend | Allowed monitor inputs |
|---|---|---|
| Step 1 robot-robot simulation | Coded simulator/RRT progress monitor | simulator state, task predicates, RRT status, done signal, timeouts |
| Step 2 human-robot simulation | Coded simulator/RRT progress monitor | same as Step 1, plus human-agent simulated state/input events |
| Step 3 real-world HRC | SARM progress monitor | camera observation, current stage definition, skill status, progress labels learned from data |

### SARM minimal output

SARM should output current-stage progress only:

```json
{
  "stage_id": "grasp_medication_bottle",
  "progress_score": 0.73,
  "is_stage_done": false,
  "message": "Object is near the gripper and partially lifted."
}
```

Failure classification is optional and should not be required for the first real-world implementation.

---

## C. Data-equivalence rule between simulation and real world

The high-level planner should receive the same type of information it would receive in the real world:

- images or perception products;
- goal;
- skill/capability descriptions;
- execution history;
- dialogue history;
- skill-level status.

The planner should not receive privileged simulator state unless the condition is explicitly labeled as privileged/oracle.

The coded simulator monitor may receive privileged simulator state because it is intentionally an oracle/coded monitor used only to test architecture value before SARM is implemented.

---

## D. Recommended module boundaries

```text
EnvironmentAdapter
    produces ObservationBundle and applies robot/human actions

PlannerInterface
    produces PlanUpdate, SkillCall, HumanInstruction, or DialogueMessage

BehaviorTreeRuntime
    manages BT ticking, local retry, and replan triggers

SkillExecutorInterface
    executes RRT or learned visual skill

ProgressMonitorInterface
    produces progress_score/stage/status from coded simulator signals or SARM

CommunicationInterface
    terminal first; speech/TTS/STT later

EpisodeLogger
    records all inputs, outputs, decisions, monitor signals, and metrics
```
