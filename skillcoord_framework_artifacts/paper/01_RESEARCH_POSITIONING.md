# 01 — Research Positioning and Contribution Claims

## Positioning statement
SkillCoord-HRC is a development and benchmarking framework for human–robot multi-agent coordination. It treats skills as typed executable units assignable to heterogeneous agents, including humans and robots. Each skill can be backed by classical planning, learned policies, scripted execution, teleoperation, simulated humans, or real humans.

## Why this matters
Current research often separates multi-agent planning from learned robot execution. A planner may generate valid symbolic allocations, but it is unclear whether those allocations remain effective when robot skills fail, humans interrupt, communication is needed, or agents have different capabilities. Conversely, learned skills are often evaluated in isolation, without testing how they affect team-level coordination.

## Contribution claims

### C1. Skill-centric coordination abstraction
A typed interface for assigning executable skills to heterogeneous human and robot agents.

### C2. Backend-independent skill execution
The same skill can be implemented by RRT, ACT, VLA, scripted controllers, teleoperation, simulated humans, or real humans.

### C3. Planner benchmark layer
A common interface for comparing centralized LLM planners, decentralized LLM agents, behavior-tree planners, symbolic planners, rule-based planners, and hybrid planners.

### C4. Failure-aware replanning
Execution outcomes are converted into structured, grounded feedback for replanning, allowing benchmarks to measure recovery rather than only clean-task success.

### C5. Human-aware evaluation
The framework evaluates task success, redundant work, idle time, communication, help requests, workload proxies, safety, and recovery.

### C6. Reproducible learned-skill integration
LeRobot-compatible data, training, inference, and benchmarking are integrated as one backend path rather than being entangled with planner logic.

## Differentiation
- Compared with RoCoBench: adds typed skill backends, learned-skill execution, human-agent modeling, and planner benchmarking beyond waypoint/motion-planning execution.
- Compared with PARTNR: focuses on executable tabletop/robot-skill coordination and backend-swappable skills.
- Compared with LeRobot: uses LeRobot for skill learning but adds multi-agent coordination, planner comparison, failure-aware replanning, and HRC metrics.
- Compared with Tool-RoCo: extends tool/agent coordination toward executable robot/human skills and learned-skill benchmarking.
