# 11 — Implementation Backlog

## Epic 1: Core schemas
AgentSpec, HumanAgentSpec, SkillSpec, SkillCall, SkillPlan, SkillResult, TeamState, TaskGoal, BenchmarkResult.

## Epic 2: Skill registry and validation
Argument validation, capability validation, preconditions, resource conflicts, workspace conflicts, skill prompt rendering.

## Epic 3: Backends
FakeSkillBackend, RRTSkillBackend, LearnedSkillBackend, ScriptedSkillBackend, SimulatedHumanBackend, TeleopBackend interface.

## Epic 4: Human model
Response delay, mistake model, unavailability, workload, communication, manual confirmation UI.

## Epic 5: Planner adapters
Rule-based, behavior tree, centralized LLM, decentralized LLM, LLM + verifier, planner-only simulator.

## Epic 6: Scheduling
Resource claims, workspace zones, sequential scheduler, concurrent scheduler, STOP_ALL cancellation, joint action merge.

## Epic 7: Monitoring and failures
Progress stages, failure taxonomy, failure injection, recovery recommendation, replanning controller, event log.

## Epic 8: Benchmark runner
Task manifests, variation manifests, baseline registry, episode runner, metrics aggregator, result schema, report generator.

## Epic 9: LeRobot integration
Dataset recorder, validator, ACT training configs, direct rollout, LearnedSkillExecutor, checkpoint registry.

## Epic 10: Paper/release
Related work table, experiment scripts, plots, environment card, dataset card, model card, tutorial, reproducibility checklist.
