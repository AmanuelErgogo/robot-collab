# Codex Orchestration Prompts

# 1. Start a phase

```text
You are implementing Phase <N> from <PHASE_FILE>.

Before editing:
1. Read 00_MASTER_ROADMAP.md, AGENTS.md, and the phase specification.
2. Inspect every completed prerequisite phase.
3. Reconcile expected and actual interfaces and list deviations.
4. Inspect the pinned LeRobot compatibility lock.
5. Identify exact files to add/change.
6. Identify schema, rollback, dependency, and backward-compatibility risks.
7. Produce a concise implementation plan.
8. Then implement the entire phase. Do not stop after planning.

Keep the change inside this phase. Never bypass a failed prerequisite.
```

# 2. Mid-implementation audit

```text
Pause and audit the current diff.

Check:
- Is the phase boundary respected?
- Did any observation/action/task schema change without a version bump?
- Did LeRobot code leak into the Python 3.8 RoCo runtime?
- Did RoCo/MuJoCo internals leak into the isolated client?
- Are success, learned success, fallback success, and overall success distinct?
- Are loops, retries, horizons, and caches bounded?
- Are failure and negative paths tested?
- Can automated tests run without a live LLM?
- Are versions, seeds, revisions, and hashes recorded?
- Are artifacts aligned to the executed transitions?

Fix every confirmed issue before continuing.
```

# 3. Completion report

```text
## Phase
## Gate decision: PASS / FAIL

## Architecture implemented
- ...

## Files added
- ...

## Files changed
- ...

## Interfaces/schema changes
- ...

## Commands run
1. `<command>`
   - exit code:
   - result:

## Tests
- passed:
- failed:
- skipped:
- blocked:

## Artifacts
- ...

## Provenance
- RoCo commit:
- LeRobot version/commit:
- Python/PyTorch/CUDA:
- bridge protocol:
- schema hash:
- dataset/checkpoint/variation revision:

## Remaining risks
- ...

## Why the gate passes or fails
- ...
```

# 4. Cross-phase final audit

```text
Audit Phases 2–8 as one system.

Trace one successful and one failed execution through:
dataset sample -> training input/output -> checkpoint -> processors -> direct rollout -> LearnedSkillExecutor -> planner feedback -> multi-agent scheduler -> benchmark result.

At every boundary verify:
- key names and ordering;
- shapes/dtypes/units;
- action representation;
- timestamps and control frequency;
- version/hash;
- reset and queue behavior;
- success/failure semantics;
- fallback attribution;
- artifact provenance.

Report findings by severity. Fix confirmed inconsistencies and rerun the smallest end-to-end test covering each boundary.
```

# 5. Security audit

```text
Audit for:
- network pickle, eval, exec;
- arbitrary remote file paths or imports;
- unbounded payloads/arrays/episodes/retries;
- credentials/tokens in configs or logs;
- exposed shutdown/debug commands;
- unchecked ndarray shape/dtype/bounds;
- unpinned checkpoint or trust_remote_code behavior;
- shell injection in training/evaluation launchers;
- unsafe local legacy-pickle migration.

Fix confirmed vulnerabilities and add regression tests.
```

# 6. Reproducibility audit

```text
Assume another researcher has only:
- repository;
- environment/lock files;
- dataset revision;
- checkpoint revision;
- benchmark version.

Determine whether they can reproduce:
1. expert data collection;
2. ACT training;
3. direct rollout;
4. LearnedSkillExecutor evaluation;
5. planner integration;
6. sequential/concurrent comparison;
7. benchmark report.

List every missing seed, config, hash, version, manifest, command, or artifact. Fix gaps and regenerate relevant documentation/cards.
```
