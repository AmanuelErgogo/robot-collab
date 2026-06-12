# AGENTS.md — RoCo × LeRobot Phases 2–8

## Runtime boundaries
- RoCo simulator stays in its validated Python 3.8 runtime unless a separate migration is approved.
- LeRobot training/inference stays in the pinned Python 3.12+ runtime.
- Use the Phase 0 bridge.
- Do not import LeRobot in normal RoCo modules.
- Do not import RoCo/MuJoCo internals in the isolated LeRobot client.

## Phase discipline
- Phase 2: replayable dataset.
- Phase 3: independently loadable ACT checkpoint.
- Phase 4: direct closed-loop rollout.
- Phase 5: production learned executor.
- Phase 6: planner and bounded replanning.
- Phase 7: safe multi-agent execution.
- Phase 8: reproducible benchmark.
Do not bypass a failed prerequisite.

## Data
- `observation[t]` is the observation before `action[t]`.
- Use simulator timestamps, not RPC latency.
- Commit only validated expert episodes.
- Split by episode/variation, never frames.
- Record schema and variation hashes.
- Never silently mix failed attempts into expert data.
- Use pinned public LeRobotDataset APIs.

## Policy
- Validate feature contracts before training/inference.
- Reset policy/action queues between episodes and skills.
- Preserve native action representation metadata.
- Never send normalized model output directly to the simulator.
- ACT first; larger VLAs only after the full ACT path works.

## Execution
- Task predicate determines success.
- Return structured evidence for failures.
- Bound steps, retries, replans, and fallbacks.
- Never hide RRT fallback.
- Keep learned, fallback, and overall success separate.
- Caller owns rollback unless the established contract explicitly differs.

## Multi-agent
- One centralized simulator step per logical tick.
- Per-agent executors must not call `env.step()` independently in concurrent mode.
- Require typed resource and workspace compatibility.
- Default to sequential when uncertain.
- Severe central safety events STOP_ALL.
- Do not claim handoff without dedicated data, implementation, and tests.

## Benchmark
- Fixed hashed variation manifests.
- Identical variations for all methods.
- Separate skill-policy and planner tracks.
- Save raw episode results.
- Core benchmark must not require a live LLM.

## Code quality
- Preserve Python 3.8 syntax in shared/server code.
- Put version-specific LeRobot code in the compatibility adapter.
- No network pickle, eval, exec, mutable defaults, or broad exception swallowing.
- Use typed models, stable errors, dependency injection, focused diffs, and negative-path tests.

## Completion report
List files, interfaces, commands, test counts, artifacts, exact versions/hashes, blockers, and remaining risks. Never claim a simulator/GPU/LLM test passed unless it ran.
