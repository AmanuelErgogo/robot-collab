# 10 — Master Coding-AI Prompt

You are implementing **SkillCoord-HRC**, a skill-centric human–robot multi-agent coordination development and benchmarking framework.

## Mission
Build a modular framework where planners assign typed executable skills to heterogeneous agents, including robots and humans. Skills can be executed by multiple backends: RRT, learned policies, scripted controllers, simulated humans, teleoperation, and real humans. The framework benchmarks planner quality, execution quality, failure recovery, human-aware coordination, and safety.

## Initial MVP
Domain: collaborative object packing / medication-style tabletop task.

Agents: Alice robot, Bob robot, Human or SimHuman.

Skills: WAIT, PICK_OBJECT, PLACE_OBJECT, PUT_OBJECT_IN_CONTAINER, HOLD_CONTAINER, READ_LABEL, CONFIRM_TARGET, ASK_FOR_HELP.

Backends: RRT, ACT learned policy, scripted, simulated human, manual/teleop later.

Planners: rule-based, behavior-tree, centralized LLM, decentralized LLM, LLM + verifier.

## Required architecture
Implement clean modules: agents, skills, planners, backends, schedulers, environments, human_models, monitors, metrics, benchmark, artifacts, configs.

## Core contracts
- Planner returns SkillPlan.
- Scheduler validates resources and decides sequential/concurrent/reject.
- Backend executes one SkillCall and returns SkillResult.
- Monitor converts raw execution into progress/failure events.
- Benchmark runner saves raw episode results and aggregate metrics.

## Rules
- Do not use free-form actions internally.
- Do not let planners choose unchecked backend internals.
- Do not silently fall back.
- Do not conflate learned success, fallback success, and task success.
- Do not use unbounded retries.
- Do not require an LLM for unit tests.
- Do not import LeRobot into simulator runtime unless explicitly supported.
- Do not use network pickle, eval, or exec.
- Use typed schemas and YAML configs.
- Add unit tests for failure paths.
- Preserve RRT as baseline and fallback.

## Implementation order
1. Typed agent and skill schemas.
2. Skill registry and validator.
3. Fake backend and benchmark smoke test.
4. RoCoBench adapter.
5. RRT backend.
6. Human-agent simulation backend.
7. Planner adapters.
8. Metrics and benchmark runner.
9. LeRobot learned-skill backend.
10. Failure injection and recovery.
11. Multi-agent scheduling/concurrency.
12. Paper/release artifacts.

## MVP definition of done
- Three planners run the same task.
- Two robot skill backends run the same skill.
- One simulated human can be assigned skills.
- Failures can be injected.
- Replanning is bounded and logged.
- Metrics include success, time, replans, redundant work, idle time, communication, safety, and recovery.
- Results are reproducible from config.
- Documentation explains how to add a skill, backend, planner, and benchmark task.
