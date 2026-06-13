# 00 — Project Charter

## Project title
**SkillCoord-HRC: A Skill-Centric Development and Benchmarking Framework for Human–Robot Multi-Agent Coordination**

## Vision
Build a modular framework where humans and robots are represented as heterogeneous agents with executable skills, enabling systematic implementation, comparison, and benchmarking of multi-agent planners under learned skills, classical skills, human participation, failures, and replanning.

## Core problem
Most systems evaluate either multi-agent planning with abstract/scripted execution or robot skill learning as isolated single-agent control. Real HRC needs both: planners must allocate tasks across humans and robots, reason about capabilities, execute skills, monitor progress, communicate, and recover from failures.

## Core thesis
A skill-centric interface separates **who decides what to do** from **how each skill is executed**, making it possible to benchmark planner methods independently from execution backends while still supporting realistic learned robot skills and human participation.

## Target users
- Robotics researchers building HRC planners.
- Robot-learning researchers evaluating learned skills in collaborative settings.
- HRI researchers studying coordination, communication, workload, and recovery.
- Developers using LLM agents, behavior trees, HTN/PDDL planners, or hybrid systems.

## First MVP domain
Collaborative object packing / medication-distribution-inspired tabletop tasks.

Agents:
- Human or SimHuman
- Alice robot
- Bob robot

Initial skills:
- WAIT()
- PICK_OBJECT(object)
- PLACE_OBJECT(object, target)
- PUT_OBJECT_IN_CONTAINER(object, container)
- HOLD_CONTAINER(container)
- READ_LABEL(object)
- CONFIRM_TARGET(object, target)
- ASK_FOR_HELP(reason)

Initial skill backends:
- RRT expert backend
- ACT learned backend
- scripted backend
- simulated-human backend
- manual/teleoperation backend later

## Main deliverables
1. Typed skill API.
2. Agent capability model.
3. Skill backend interface.
4. Human-agent simulation model.
5. Planner interface and adapters.
6. Benchmark task suite.
7. Failure injection and monitoring.
8. Metrics and result schema.
9. LeRobot integration for learned skills.
10. Paper-ready evaluation and baselines.

## MVP success definition
The MVP is successful if it can run the same benchmark task using multiple planners and multiple execution backends, produce standardized metrics, and demonstrate that planner quality can be evaluated separately from learned-skill success.
