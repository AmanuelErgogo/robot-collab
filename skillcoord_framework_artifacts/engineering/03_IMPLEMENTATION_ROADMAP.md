# 03 — Implementation Roadmap

## Phase A — Foundation and interfaces
Deliver agent model, skill model, SkillRegistry, SkillBackend interface, planner interface, outcome schema, metric schema, and YAML templates.

Exit: define agents/skills in config; validate a SkillPlan without executing; run fake backends in tests.

## Phase B — RoCoBench adapter
Deliver RoCo adapter, object/agent state summarizer, resource/workspace extraction, RRT backend wrapper, execution logs, reset/variation handling.

Exit: typed SkillPlan compiles to existing RoCo execution and RRT backend succeeds on a simple task.

## Phase C — LeRobot learned-skill integration
Deliver recorder, LeRobotDataset writer/validator, ACT training config, direct rollout, LearnedSkillExecutor, checkpoint registry.

Exit: one learned skill runs in closed loop and learned success/failure is separated from fallback.

## Phase D — Human-agent model
Deliver simulated-human backend, delay model, mistake model, unavailability model, communication actions, optional manual confirmation UI.

Exit: planner can assign skills to a human agent and human workload/idle metrics are computed.

## Phase E — Planner adapters
Deliver centralized LLM, decentralized LLM, behavior-tree, rule-based, optional HTN/PDDL, output verifier, bounded replanning.

Exit: at least three planners run the same benchmark.

## Phase F — Benchmark suite
Deliver task manifests, variation manifests, failure-injection scenarios, benchmark runner, metrics report, baselines, result schema.

Exit: same variations used across planners/backends and raw episode results are saved.

## Phase G — Paper and release
Deliver paper draft, related-work matrix, cards, README, tutorial, ablations, release checklist.

Exit: code and artifacts reproduce main results.

## Suggested 12-week MVP timeline

- Weeks 1–2: core schemas, registry, fake backend, tests.
- Weeks 3–4: RoCo adapter and RRT skill backend.
- Weeks 5–6: dataset generation and ACT skill training.
- Weeks 7–8: learned executor and human simulation backend.
- Weeks 9–10: planner adapters, failure injection, metrics.
- Weeks 11–12: benchmark runs, plots, paper draft, release docs.
