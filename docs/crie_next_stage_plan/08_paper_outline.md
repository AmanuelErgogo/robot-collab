# Paper Outline

## Working title

**CRIE-BT: Behavior-Tree-Mediated Visual-Language Coordination with Explicit Progress Monitoring for Human-Robot Collaboration**

Alternative shorter title:

**Explicit Progress Monitoring Improves VLM-Based Coordination in Human-Robot Collaboration**

## Core thesis

VLM-based agents can decompose collaborative tasks and produce plausible next actions, but monolithic VLM planners are weak execution monitors. They must infer progress, failure, and recovery needs inside the same slow language loop. CRIE-BT separates high-level reasoning, behavior-tree execution, and progress monitoring. This modular design enables fast local recovery, explicit replanning triggers, and a clean path from simulation with coded monitors to real-world human-robot collaboration with SARM progress monitoring.

## Abstract structure

1. Problem: VLM/VLA planners help with task reasoning but struggle with execution-level progress monitoring and recovery in collaborative settings.
2. Gap: Existing monolithic planners often self-monitor from image feedback without an explicit progress module, making recovery slow, unstable, or hard to evaluate.
3. Method: CRIE-BT separates planning, BT-mediated execution, and progress monitoring.
4. Staging: simulation uses RRT skills and coded simulator monitor; real-world version uses learned visual skills and SARM progress monitoring.
5. Evaluation: robot-robot simulation, human-robot simulation, and real human-robot tasks in medication dispensing and cooking.
6. Result claim placeholder: CRIE-BT improves task success, recovery, and replanning efficiency over VLM self-monitoring baselines.

## Introduction outline

### Paragraph 1: Motivation

Collaborative robots must coordinate with other robots and humans in partially observable, dynamic scenes. They need to decide who should do what, communicate intent, and recover when execution deviates from the plan.

### Paragraph 2: VLM/VLA opportunity and limitation

VLMs and VLAs offer visual reasoning and language communication, but a monolithic planner is often asked to do too much: task decomposition, allocation, progress checking, failure detection, and replanning.

### Paragraph 3: Why self-monitoring is insufficient

In collaborative manipulation, progress is not only whether the scene changed. A partner may be delayed, an object may be partially moved, a robot may be stuck, or a human may counter-propose. Image-only self-monitoring can confuse these cases, leading to unnecessary replans or missed recovery opportunities.

### Paragraph 4: Proposed approach

CRIE-BT separates high-level VLM reasoning from execution control and progress monitoring. The planner proposes or updates a behavior tree; the BT manages execution; an explicit monitor tracks current-stage progress and triggers retry or replanning.

### Paragraph 5: Staged evaluation

The system is evaluated in three stages: robot-robot simulation with RRT and coded monitor, human-robot simulation with terminal human control, and real human-robot collaboration with learned visual skills and SARM progress monitoring.

### Paragraph 6: Contributions

Possible contribution list:

1. A modular CRIE-BT architecture for VLM-based collaborative task coordination.
2. A unified condition taxonomy covering centralized/dialog and robot-robot/human-robot teams.
3. A staged evaluation protocol that isolates architecture value before replacing coded simulator monitors with SARM learned progress monitors.
4. Real-world validation on collaborative medication dispensing and cooking tasks.

## Related work sections

1. VLM/VLA planning for embodied agents.
2. Behavior trees and structured robot task execution.
3. Progress monitoring and reward modeling.
4. Human-robot collaboration, dialogue, and shared autonomy.
5. Simulation-to-real evaluation of collaborative manipulation.

## Method section outline

1. Problem formulation.
2. Condition taxonomy.
3. Baseline: VLM self-monitoring.
4. CRIE-BT architecture.
5. Progress monitor backends:
   - coded simulator monitor;
   - SARM progress monitor.
6. Human interface and communication.
7. Implementation details.

## Experiment section outline

1. Step 1: robot-robot simulation.
2. Step 2: human-robot simulation with terminal interface.
3. Step 3: real human-robot tasks.
4. Metrics.
5. Ablations.

## Results section outline

1. Main comparison table across conditions.
2. Per-task success and time.
3. Recovery and replanning behavior.
4. Monitor quality and detection latency.
5. Human-robot coordination analysis.
6. Qualitative examples.

## Limitations section

- Step 1 and Step 2 use privileged coded simulator monitor for CRIE-BT.
- SARM must be trained and validated for real task stages.
- Terminal human interface is simpler than natural speech and full physical collaboration.
- Real-world learned skills may introduce failures that are not captured in simulation.

## Claim boundaries

Do not write:

> We show that SARM improves simulation performance.

Instead write:

> Simulation experiments isolate the architectural benefit of explicit progress monitoring using a coded/oracle monitor; real-world experiments evaluate replacing that monitor with SARM.
