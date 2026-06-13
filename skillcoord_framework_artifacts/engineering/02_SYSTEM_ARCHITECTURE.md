# 02 — System Architecture

## High-level architecture

```text
Task Goal
  ↓
Planner Interface
  ↓
Typed Multi-Agent SkillPlan
  ↓
Semantic Validator
  ↓
Resource / Workspace Scheduler
  ↓
Skill Executor Router
  ↓
Skill Backends
  ├── RRT backend
  ├── ACT / Diffusion / VLA backend
  ├── scripted backend
  ├── simulated-human backend
  ├── teleoperation backend
  └── real-human backend
  ↓
Environment / Robot Interface
  ↓
Outcome Monitor
  ↓
Structured Feedback
  ↓
Replanning / Recovery
  ↓
Benchmark Metrics
```

## Core abstractions

### Agent
An entity that can be assigned skills. Types: robot, human, simulated human, virtual agent, tool agent.

Fields: name, type, embodiment, reachable workspace, capabilities, availability, cost, reliability model, communication model, safety constraints.

### Skill
A typed executable action, such as `PUT_OBJECT_IN_CONTAINER(object, container)`, `READ_LABEL(object)`, or `ASK_FOR_HELP(reason)`.

Fields: name, arguments, preconditions, postconditions, resources, failure modes, supported agents, allowed backends, expected duration, communication needs.

### Skill backend
A concrete implementation of a skill, such as RRT, learned ACT, scripted, simulated human, teleop, or real human.

### Planner
Maps task state and goal into a typed SkillPlan.

### Scheduler
Checks resource/workspace conflicts and determines sequential or concurrent execution.

### Outcome monitor
Turns raw execution into structured outcomes: success, timeout, missed grasp, slippage, human unavailable, redundant retrieval, passive wait, safety conflict, etc.

## Runtime modes

1. Planner-only benchmark: simulated skill outcomes.
2. Skill-execution benchmark: benchmark supplies skill calls.
3. Planner + skill benchmark: planner chooses skills and real backends execute.
4. Human-in-the-loop benchmark: real or simulated human participates.

## Recommended initial system boundary
Version 1 should support one tabletop collaborative domain, two robots and one human model, 5–8 typed skills, RRT + ACT + simulated human backends, multiple planner baselines, failure injection, and recovery evaluation.
