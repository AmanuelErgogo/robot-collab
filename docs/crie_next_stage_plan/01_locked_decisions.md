# Locked Decisions Before Implementation

## 1. Baseline planner input

The baseline receives:

- goal;
- agent capabilities;
- current image or visual observation;
- previous actions;
- dialogue/history in dialog conditions;
- execution status from the previous skill, such as success, failure, timeout, or `done`, if that is how the simulator reports execution progress/status.

The baseline should not receive privileged structured simulator state, object poses, reward internals, or coded progress labels unless the experiment is explicitly labeled as an oracle/text-state ablation.

## 2. CRIE-BT coded monitor input in simulation

For simulator experiments, the coded monitor may receive any proposed simulator/task signal that can reveal whether the current subtask is processing, complete, stuck, or failed. This can include:

- simulator object poses or symbolic task predicates;
- RRT execution status;
- `env.get_reward_done()` or task `done` signal;
- collision, timeout, grasp, placement, or path-planning status;
- task-specific predicates for medication dispensing and cooking.

This monitor is privileged. The paper should call it a coded/oracle simulator progress monitor.

## 3. Shared skill backend in Step 1 and Step 2

Both baseline and CRIE-BT conditions use RRT skill execution in Step 1 and Step 2. This makes the comparison primarily about:

- monolithic VLM self-monitoring versus explicit progress monitoring;
- direct action-loop control versus BT-mediated execution;
- centralized versus dialog/distributed coordination;
- robot-robot versus human-robot teams.

## 4. Centralized human-robot condition

A single central planner sees the shared observation and assigns work to both robot and human.

- Robot receives skill calls.
- Human receives natural-language instructions through the terminal first, and later speech.
- Human may follow, delay, reject, counter-propose, or make mistakes.

## 5. Dialog/distributed human-robot condition

The robot has its own planner/controller and coordinates with an independent human collaborator through dialogue.

- There is no central planner controlling the human.
- The robot can propose actions, ask for help, acknowledge human actions, or revise its own plan.
- The human is controlled through a terminal/keyboard interface first.

## 6. Human interface staging

Step 2 should begin with a terminal/keyboard human interface. Speech/TTS/STT should be added later as a wrapper around the same communication interface.

## 7. Real-world tasks

Step 3 real-world tasks:

1. Collaborative medication dispensing.
2. Collaborative cooking.

## 8. SARM output

SARM is a progress monitor, not necessarily a failure classifier. The required output is:

- current stage;
- progress score for the current stage, from 0 to 1;
- optionally confidence, notes, or evidence.

A failure classifier can be added later, but it is not required for the core Step 3 claim.

## 9. Condition names

The eight condition names are:

- `VLM-RR-Cent`
- `VLM-RR-Dialog`
- `VLM-HR-Cent`
- `VLM-HR-Dialog`
- `CRIE-BT-RR-Cent`
- `CRIE-BT-RR-Dialog`
- `CRIE-BT-HR-Cent`
- `CRIE-BT-HR-Dialog`

Backends are reported separately:

- skill backend: RRT / learned visual skill;
- monitor backend: VLM self-monitor / coded simulator monitor / SARM monitor;
- environment: simulation / real;
- team: robot-robot / human-robot.

## 10. Paper claim discipline

- Step 1 proves architecture value in robot-robot simulation with RRT and coded monitor.
- Step 2 tests transfer of the same architecture to human-robot simulation with a terminal human interface.
- Step 3 tests real human-robot collaboration with learned visual skills and SARM progress monitoring.

Do not claim that Step 1 or Step 2 proves learned SARM monitoring. They use a coded/oracle simulator monitor.
