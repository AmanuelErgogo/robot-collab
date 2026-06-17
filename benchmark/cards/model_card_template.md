# Model Card Template: RoCoBench-Pack-Skills-v1

## Model

- Method:
- Checkpoint:
- Training config:
- Dataset revision:
- Benchmark version:
- Variation manifest hash:

## Intended Use

Report either the `skill_policy` track or the `planner_skill` track. Do not mix track scores.

## Results

- Raw run directory:
- `episodes.csv`:
- `metrics.json`:
- Success rate:
- Wilson 95 percent interval:
- Failure distribution:

## Learned And Fallback Accounting

- Learned attempts:
- Learned successes:
- Fallback attempts:
- Fallback successes:
- Overall successes:

Fallback success must not be counted as learned success.

## Reproducibility

Include:

- RoCo commit:
- LeRobot version and commit:
- bridge protocol:
- schema hash:
- action representation:
- dataset/checkpoint revision:
- command line:

## Limitations

Document unsupported cameras, objects, target slots, runtime dependencies, and any blocked bridge/simulator/GPU checks.

