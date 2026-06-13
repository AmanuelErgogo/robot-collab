# 07 — Evaluation Protocol

## Research questions

RQ1. How do centralized LLM, decentralized LLM, behavior-tree, and rule-based planners compare on multi-agent skill allocation?

RQ2. How does planner performance change when RRT skills are replaced by learned ACT skills?

RQ3. Does explicitly modeling humans as agents improve workload, idle time, redundancy, and recovery?

RQ4. Which planner methods recover most effectively from skill failures and human interruptions?

RQ5. Does the typed skill interface improve validity, safety, and reproducibility compared with free-form planner outputs?

## Experimental conditions

Planner conditions:
1. Rule-based planner
2. Behavior-tree planner
3. Centralized LLM planner
4. Decentralized LLM-agent planner
5. LLM + verifier

Backend conditions:
1. All RRT
2. RRT + simulated human
3. ACT robot skills + simulated human
4. ACT + RRT fallback
5. Optional real human or teleoperation

Failure conditions:
1. No injected failure
2. Robot skill failure
3. Human delay/unavailability
4. Human wrong action
5. Target conflict
6. Safety conflict

## Minimum MVP matrix

```text
3 planners × 3 backend settings × 3 failure settings × N variations
```

Recommended N:
- Debug: 5 variations
- Pilot: 20 variations
- Paper: 50–100 variations if runtime permits

## Primary outcomes
- task success rate
- recovery success
- replan count
- redundant work
- passive wait
- safety violations
- learned skill success
- fallback rate
- completion time

## Statistical analysis
- Wilson confidence intervals for success rates
- paired comparisons using identical variations
- paired bootstrap for completion time
- McNemar-style paired tests for binary success when appropriate
- effect sizes and confidence intervals
- raw episode CSV for reproducibility

## Ablations
1. Without skill validator
2. Without failure feedback
3. Without human-agent model
4. Without fallback
5. Free-form action output vs typed skill output
6. Learned-only vs learned + fallback
7. Centralized vs decentralized planning

## Paper-ready tables

Table 1: Related-framework comparison.
Table 2: Planner results.
Table 3: Backend results.
Table 4: Failure recovery by failure type.

## Plots
- success rate by planner
- recovery success by failure type
- redundant work distribution
- completion time boxplot
- fallback stacked bar
- planner validity/error categories
- coordination timeline examples
