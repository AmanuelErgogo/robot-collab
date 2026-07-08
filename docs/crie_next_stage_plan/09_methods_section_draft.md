# Draft Method Section

## Method Overview

We study collaborative manipulation settings in which a team must complete a shared goal through coordinated subtasks. The team may contain two robots or one robot and one human collaborator. We compare monolithic VLM-based coordination baselines against CRIE-BT, a modular controller that separates high-level visual-language reasoning, behavior-tree-mediated execution, and explicit progress monitoring.

The key distinction is how execution progress is represented. In the baseline, the VLM planner receives image feedback, action history, dialogue history when available, and the previous execution status. The same VLM loop is responsible for task decomposition, allocation, progress assessment, and replanning. In CRIE-BT, the planner is not the only execution monitor. A behavior tree executes the current plan, while an explicit progress monitor estimates the progress of the current stage and triggers continuation, local retry, or replanning.

## Problem Formulation

Let a collaborative task be specified by a goal `G`, a set of agents `A`, a set of available skills `K`, and a stream of visual observations `O_t`. At each decision point, the system selects either a robot skill call, a human instruction, or a dialogue message. A robot skill call is executed by a low-level backend such as RRT in simulation or a learned visual skill in the real world. A human instruction is communicated to the human collaborator through a terminal interface in simulation and optionally speech in later deployments.

The controller must solve three coupled problems: task decomposition, subtask allocation, and execution recovery. Recovery is required when the current subtask fails, stalls, becomes unnecessary, or conflicts with the collaborator's behavior.

## Baseline: VLM Self-Monitoring

The baseline uses a monolithic VLM planner. The planner receives the task goal, agent capabilities, current image observation, previous actions, dialogue history in dialog conditions, and the previous execution status. It then directly outputs the next robot skill call, human instruction, or dialogue message. The baseline has no separate progress-monitor module. Progress and failure are inferred inside the VLM planning loop from visual feedback and execution status.

We evaluate four baseline conditions: centralized robot-robot coordination, dialog robot-robot coordination, centralized human-robot coordination, and dialog human-robot coordination. In centralized conditions, one planner assigns work to all collaborators. In dialog/distributed conditions, each robot-side agent plans from its own perspective and coordinates through dialogue.

## CRIE-BT

CRIE-BT decomposes the controller into three modules. First, a VLM-based CRIE planner proposes or updates a behavior-tree representation of the collaborative plan. Second, a behavior-tree runtime ticks the plan and dispatches robot skill calls or human instructions. Third, an explicit progress monitor estimates the progress of the active stage and determines whether the controller should continue, locally retry, mark the stage complete, or request a replan.

This decomposition reduces the burden on the VLM planner. The VLM is used for high-level reasoning, allocation, and language generation, while execution-level progress checking is handled by a dedicated monitor. The behavior tree provides a structured execution substrate for local retries, fallback behavior, and controlled replanning.

## Progress Monitor Backends

CRIE-BT uses different monitor backends across the development stages.

In simulation, CRIE-BT uses a coded simulator/RRT progress monitor. This monitor may access simulator and task signals such as object poses, grasp state, placement predicates, RRT success or failure, timeout, and task `done`. Because these signals are privileged, the simulation monitor is used to isolate the value of the architecture rather than to claim learned perception-based monitoring.

In the real-world system, the coded monitor is replaced by SARM, a Stage-Aware Reward Modeling progress monitor. SARM estimates a scalar progress score for the current task stage, where 0 indicates no progress and 1 indicates that the current stage is complete. The minimal SARM output is the current stage identifier, progress score, and a stage-completion decision. Failure classification is optional and not required for the core controller.

## Coordination Modes

We evaluate centralized and dialog/distributed coordination. In centralized coordination, one planner/controller assigns actions to all agents. For human-robot teams, this means that the central controller sends robot skill calls to the robot and natural-language instructions to the human.

In dialog/distributed coordination, each robot-side agent has its own planner/controller and coordinates through dialogue. In human-robot teams, the robot does not control the human; it proposes, asks, acknowledges, and adapts to the human's responses.

## Human Interface

In the first human-robot simulation stage, the human collaborator controls one simulated agent through a terminal interface. The interface supports accepting an instruction, rejecting it, counter-proposing, marking a requested action as done, and controlling the simulated human-side agent. The same communication interface can later be wrapped with speech recognition and text-to-speech without changing the controller logic.

## Fairness and Information Access

To ensure fair comparison, the planner in both baseline and CRIE-BT receives only information that would be available in the real world: images or public percepts, task goal, capabilities, action history, dialogue history, and skill-level execution status. Privileged simulator state is reserved for the coded simulator progress monitor and evaluation. Runs that provide privileged state to a planner must be reported separately as oracle ablations.

## Experimental Conditions

The eight primary conditions are `VLM-RR-Cent`, `VLM-RR-Dialog`, `VLM-HR-Cent`, `VLM-HR-Dialog`, `CRIE-BT-RR-Cent`, `CRIE-BT-RR-Dialog`, `CRIE-BT-HR-Cent`, and `CRIE-BT-HR-Dialog`. In simulation, both baseline and CRIE-BT use the same RRT skill backend. In real-world experiments, both use learned visual skills, while CRIE-BT uses SARM for explicit progress monitoring.
