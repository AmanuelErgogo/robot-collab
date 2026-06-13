# 08 — Paper Skeleton

## Working title
**SkillCoord-HRC: A Skill-Centric Framework for Benchmarking Human–Robot Multi-Agent Coordination**

## Abstract draft
Human–robot collaboration requires more than isolated robot skill execution: agents must allocate subtasks, coordinate timing, communicate intent, monitor progress, and recover when either humans or robots fail. Existing benchmarks often emphasize either multi-agent planning with abstract execution or robot skill learning in single-agent settings. We introduce SkillCoord-HRC, a modular development and benchmarking framework that represents humans and robots as heterogeneous agents with typed executable skills. A planner outputs skill assignments, while each skill can be executed by interchangeable backends including classical motion planning, learned policies, scripted controllers, simulated humans, teleoperation, or real humans. The framework provides semantic validation, resource-aware scheduling, failure monitoring, grounded replanning feedback, and standardized metrics for task success, redundancy, idle time, safety, communication, and recovery. We instantiate the framework in collaborative tabletop tasks and evaluate multiple planner families and skill backends under clean and failure-injected conditions.

## Introduction outline
1. Real HRC requires coordination under heterogeneous capabilities and failures.
2. Planning benchmarks and skill-learning frameworks are disconnected.
3. Planners need executable skills, and learned skills need team-level evaluation.
4. SkillCoord-HRC provides a typed skill-centric framework.
5. Contributions: abstraction, backend-swappable execution, failure-aware replanning, benchmark tracks, RoCoBench/LeRobot implementation.
6. Experimental summary.

## Method section
1. Agent and skill model.
2. Skill backend abstraction.
3. Planner interface.
4. Validation and scheduling.
5. Execution monitoring and feedback.
6. Benchmark tracks and metrics.

## Experiments section
- Tasks: collaborative packing / medication-style tabletop coordination.
- Planners: rule-based, behavior tree, centralized LLM, decentralized LLM, LLM + verifier.
- Backends: RRT, ACT, ACT + fallback, simulated human.
- Metrics: task success, replans, redundant work, idle time, communication, safety, recovery.
- Ablations: no validator, no feedback, no human model, no fallback, free-form vs typed.

## Limitations
- Initial tasks are tabletop and simplified.
- Human model is approximate.
- Learned skill performance depends on dataset quality.
- Real-human studies require separate IRB and may differ from simulated-human results.
- Concurrency support is conservative.
- Real-world transfer requires robot-specific safety validation.

## Future work
- Real-human experiments.
- Richer household domains.
- Learned human behavior models.
- Multi-robot joint policies.
- Natural-language explanation evaluation.
- Real-time interruptible behavior-tree execution.
