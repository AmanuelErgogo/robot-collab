# 04 — Engineering Requirements

## Functional requirements

FR1. The framework shall represent planner outputs as typed skill calls, not free-form low-level text.

FR2. The framework shall support robots, simulated humans, real humans, and tool agents.

FR3. A skill shall be executable by multiple backends without changing planner output.

FR4. The framework shall validate known skill, known arguments, agent capability, object existence, target availability, preconditions, resource conflicts, and workspace conflicts.

FR5. Skill execution shall return structured outcomes with failure code, evidence, progress stage, backend, and recovery recommendation.

FR6. Centralized, decentralized, symbolic, behavior-tree, and hybrid planners shall run through the same planner interface.

FR7. Every benchmark run shall record seeds, variations, planner config, skill backend config, versions, and raw episode metrics.

FR8. The framework shall compute human-aware metrics: workload proxies, communication, idle time, redundant work, help requests, and safety.

## Non-functional requirements

NFR1. Modularity: planner, skill API, backend execution, scheduler, environment, and metrics remain separable.

NFR2. Reproducibility: experiments rerun from config and manifests.

NFR3. Testability: core logic unit-testable without MuJoCo, LeRobot, GPU, or LLM APIs.

NFR4. Safety: no network pickle, eval, arbitrary command execution, unbounded retries, or unchecked actions.

NFR5. Backward compatibility: RRT baseline and previous RoCo-style execution remain available.

## Minimum public MVP

- One domain.
- Two robots and one human model.
- At least five skills.
- At least three planner baselines.
- RRT and ACT backends.
- Failure injection.
- Standardized metrics.
- Reproducible benchmark report.
