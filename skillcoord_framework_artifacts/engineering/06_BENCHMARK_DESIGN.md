# 06 — Benchmark Design

## Benchmark name
**SkillCoord-HRC-Bench**

## Benchmark tracks

### Track 1: Planner-only
Planner outputs typed skill plans. Skill outcomes are simulated.

### Track 2: Skill execution
Benchmark provides skill calls. Backends execute them.

### Track 3: Planner + execution
Planner chooses skills and actual backends execute them.

### Track 4: Human-in-the-loop
Human is real or simulated.

### Track 5: Failure recovery
Failures are injected and planners must recover.

## Initial tasks

### Task A: Collaborative grocery packing
Challenges: duplicate assignment, target conflict, robot reach limits, parallel vs sequential work, human confirmation.

### Task B: Medication sorting
Challenges: label reading, target correctness, safety-critical confirmation, recovery after wrong item.

### Task C: Assisted handover
Challenges: timing, waiting, handoff failure, interruption.

### Task D: Shared cleanup
Challenges: division of labor, redundant retrieval, human availability, robot failure.

## Failure injection scenarios
- robot missed grasp
- object slip
- target occupied
- duplicate assignment
- human unavailable
- human wrong-object action
- robot timeout
- safety zone conflict
- perception loss
- planner invalid action
- communication delay

## Planner baselines
- rule-based planner
- behavior-tree planner
- centralized LLM planner
- decentralized LLM-agent planner
- LLM + verifier
- optional HTN/PDDL planner
- original RoCo-style baseline

## Skill backend baselines
- RRT expert
- scripted controller
- ACT learned skill
- VLA learned skill later
- simulated human
- no-op/hold
- bounded random control

## Metrics

Task metrics: success, completion time, steps, replans, recovery success.

Coordination metrics: redundant work, passive wait, duplicate claims, target conflicts, human idle time, robot idle time, communication count.

Safety metrics: safety-zone violations, collision/near miss, unsafe handover, wrong target delivery.

Human-aware metrics: workload proxy, help requests, unnecessary human involvement, communication clarity, trust/fluency survey if real humans are used.

Execution metrics: skill success, learned skill success, fallback use, RRT fallback success, timeout, slippage, missed grasp.
