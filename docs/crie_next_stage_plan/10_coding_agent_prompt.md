# Coding Agent Prompt

You are implementing the next stage of the CRIE-BT/SARM human-robot collaboration project. Use the uploaded repo as the starting point.

## Research framing to preserve

There are two method families:

1. **Baseline VLM self-monitoring**: a monolithic VLM planner receives image/status/history/dialogue feedback and directly decides the next action. It has no separate progress monitor.
2. **CRIE-BT**: a VLM planner creates/updates a behavior tree; the BT executes actions; an explicit progress monitor tracks current-stage progress and triggers continue/retry/replan.

Do not label the baseline as SARM. SARM belongs to the real-world CRIE-BT monitor backend.

## Conditions to support

Implement a condition registry with these eight paper names and code names:

- `VLM-RR-Cent` / `vlm_rr_cent`
- `VLM-RR-Dialog` / `vlm_rr_dialog`
- `VLM-HR-Cent` / `vlm_hr_cent`
- `VLM-HR-Dialog` / `vlm_hr_dialog`
- `CRIE-BT-RR-Cent` / `criebt_rr_cent`
- `CRIE-BT-RR-Dialog` / `criebt_rr_dialog`
- `CRIE-BT-HR-Cent` / `criebt_hr_cent`
- `CRIE-BT-HR-Dialog` / `criebt_hr_dialog`

Backends are separate from condition names:

- skill backend: `RRT` or `LearnedSkill`;
- monitor backend: `VLM-self`, `CodedSim`, or `SARM`;
- environment: `sim` or `real`.

## First implementation target: Step 1

Implement robot-robot simulation conditions with RRT skills:

- `VLM-RR-Cent`
- `VLM-RR-Dialog`
- `CRIE-BT-RR-Cent`
- `CRIE-BT-RR-Dialog`

For CRIE-BT, use a coded simulator/RRT progress monitor. It may use simulator/task signals including object poses, task predicates, RRT status, timeout, and done signal.

For baseline, do not instantiate a separate monitor. The VLM planner self-monitors from image/status/history feedback.

## Second implementation target: Step 2A

Implement human-robot simulation conditions with a terminal human interface:

- `VLM-HR-Cent`
- `VLM-HR-Dialog`
- `CRIE-BT-HR-Cent`
- `CRIE-BT-HR-Dialog`

One collaborator is controlled by a human through terminal/keyboard commands. Keep speech out of the first implementation; later add speech as a wrapper around the same communication interface.

## Interface requirements

Create or align these interfaces:

- `ObservationBundle`
- `SkillCall`
- `ExecutionFeedback`
- `MonitorDecision`
- `EnvironmentAdapter`
- `PlannerInterface`
- `SkillExecutorInterface`
- `ProgressMonitorInterface`
- `CommunicationInterface`
- `CollaborationController`

Add a hard function to strip privileged simulator state before planner prompts:

```python
def planner_safe_observation(obs):
    # remove oracle_state before passing to any non-oracle planner
    ...
```

Only the coded simulator monitor and evaluator may consume `oracle_state`.

## Naming cleanup

The current repo has names like `vlm_sarm_monitor.py` and `SimulatorSignalVLMSARMMonitor`. These are confusing because the baseline does not use SARM. Rename or wrap them as:

- `CodedSimProgressMonitor`
- `SARMProgressMonitor` only for the real-world learned monitor
- `NoSeparateMonitor` or `VLM-self` only as a logging placeholder for baseline

Keep backward-compatible aliases if existing tests depend on the old names.

## Logging requirements

Every episode row must include:

- `condition_name`
- `controller_family`
- `team_type`
- `coordination_mode`
- `skill_backend`
- `monitor_backend`
- `monitor_privileged`
- `environment`
- `planner_input_type`
- `task_id`
- `success`
- `task_done`
- `completion_time_s`
- `planner_calls`
- `replans`
- `local_retries`
- `failed_subtasks`
- `dialogue_turns`
- `human_interventions`
- `monitor_updates`
- `wall_time_s`

Also write `events.jsonl` per episode with planner calls, skill starts, skill feedback, monitor updates, replans, human instructions/responses, and dialogue messages.

## Tests to add

Add tests for:

1. all eight conditions load from the registry;
2. baseline conditions have no explicit monitor;
3. CRIE-BT Step 1/2 conditions use `CodedSimProgressMonitor`;
4. Step 3 CRIE-BT conditions use `SARMProgressMonitor`;
5. `planner_safe_observation` strips `oracle_state`;
6. terminal human interface returns accept/reject/counter/done responses;
7. episode logs contain all required metadata;
8. monitor progress score is clamped to `[0,1]`.

## Do not overclaim

Simulation CRIE-BT uses a coded/oracle monitor. Do not write code comments, logs, or reports implying that Step 1/2 evaluates learned SARM. SARM is Step 3.
