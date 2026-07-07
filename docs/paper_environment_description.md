# Environment and Evaluation Description

This document describes the current paper-facing evaluation scope for CRIE-BT
in RoCoBench simulation and the remaining real-world transfer boundary.

## Tasks and Team Configurations

The simulator paper scope is Robot-Robot, meaning all task agents are
autonomous robot agents controlled through the RoCoBench `EXECUTE / NAME /
ACTION` interface. The four paper tasks are:

| Task ID | Paper name | Simulator class |
| --- | --- | --- |
| `sandwich` | Sandwich | `MakeSandwichTask` |
| `pack` | Pack Grocery | `PackGroceryTask` |
| `cabinet` | Cabinet | `CabinetTask` |
| `sort` | Sort | `SortOneBlockTask` |

The simulator runner also supports `sweep` and `rope`, but those are outside
the four-task paper scope. Cabinet and Sort can involve more than two robot
agents in the underlying task definition, so the paper should use
"robot-only multi-robot" rather than "two-robot" for the full simulator set.

The real-world paper scope remains collaborative medication dispensing and
collaborative cooking. Learned policies are real-world components only until a
CRIE-BT executor wrapper is added.

## Primary Methods

Only two methods are primary paper methods:

| Paper method | Runner flags | Implementation |
| --- | --- | --- |
| CRIE-BT-Dialog | `--mode bt_mediated --planner-mode dialog --adapter legacy` | Per-agent RoCoBench dialogue prompter plus BT-mediated execution and recovery. |
| VLM/SARM-Monitor-Planner-Dialog | `--mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy` | Per-agent RoCoBench dialogue prompter plus a VLM/SARM monitor-planner interface. In simulation, the backend uses simulator done and executor failed signals. |

Centralized ablations:

| Ablation | Runner flags | Purpose |
| --- | --- | --- |
| CRIE-BT-Cent | `--mode bt_mediated --planner-mode chat --adapter legacy` | Centralized planner with BT-mediated recovery. |
| VLM/SARM-Monitor-Planner-Cent | `--mode vlm_sarm_monitor_planner --planner-mode chat --adapter legacy` | Centralized planner with the simulator-backed VLM/SARM monitor interface. |

`direct_feedback` and `open_loop` remain available for debugging and legacy
comparisons, but they are not part of the current paper method set.

The reported VLM/SARM simulator baseline replaces the old direct-feedback
baseline in the paper. In simulation, `SimulatorSignalVLMSARMMonitor` consumes
the same done/failure facts that direct feedback uses implicitly, so
`VLM/SARM-Monitor-Planner-Dialog` should be treated as the dialog direct-
feedback-equivalent baseline with monitor-interface logging.

## VLM/SARM Simulator Backend

The VLM/SARM simulator baseline is implemented through
`BaseVLMSARMMonitorBackend.evaluate(...)`. This is the same interface expected
from the real VLM/SARM backend. In simulator runs, `SimulatorSignalVLMSARMMonitor`
fills the interface by mapping:

| Real monitor concept | Simulator source |
| --- | --- |
| `DONE` | `env.get_reward_done(observation)[1] == True` |
| `FAILED` | executor feedback has `failure.is_failure` or `status == FAILURE` |
| `IN_PROGRESS` | neither done nor failed |

The baseline does not use the CRIE-BT behavior-tree retry policy. On `FAILED`,
it requests a new planner action through the same planner contract.

## Metrics

Episode rows from `scripts/run_crie_bt_sim.py` contain the metrics needed for
implemented simulator runs:

| Metric | Source field | Implementation status |
| --- | --- | --- |
| Success rate | `sim_success` | Task-completion success from `env.get_reward_done()`. |
| Controller success | `success` | Whether the controller completed the episode. |
| Completion time | `wall_time_s` | Wall-clock time around one controller episode. |
| Token consumption | `llm_prompt_tokens`, `llm_completion_tokens`, `llm_total_tokens` | Aggregated from per-call LLM `usage` returned by the prompter. |
| LLM latency | `llm_call_latencies_s` | Per-call latency measured at the prompter boundary. |
| Recovery behavior | `replans`, `local_retries`, `failure_counts`, `events` | Logged by both primary methods. |
| Monitor decisions | `VLM_SARM_MONITOR` events | Logged by the VLM/SARM monitor-planner baseline. |
| Reactivity | `reactivity_s` when annotated | Analyzer-supported, but not automatically measured from simulator events yet. |
| Hallucination rate | `hallucination_count` / `hallucination_annotation_count` | Analyzer-supported for manual annotations; no automatic hallucination judge is implemented. |

`scripts/analyze_crie_bt_eval.py` writes `summary.csv` and `summary.md`, and
can group rows by mode, task/mode, or task/method.

## Result Paths

Use the following canonical root for new Robot-Robot simulator results:

```text
results/robot_robot_sim_v1/{task_id}/{method}/episodes.jsonl
results/robot_robot_sim_v1/{task_id}/{method}/analysis/
results/robot_robot_sim_v1/{task_id}/{method}/prompts/
```

Method directory names should be lowercase and stable:

```text
crie_bt_dialog
vlm_sarm_dialog
crie_bt_cent
vlm_sarm_cent
```

Older paths such as `results/sandwich_bt_mediated/` and
`results/c3_to_c6_runs/` are legacy artifacts and should not be used for new
paper runs.

## Real-World Transfer Blockers

The simulator implementation already uses the high-level interfaces needed for
real deployment: `CollaborativePlan`, `SkillCall`, `ExecutionFeedback`,
`BaseSkillExecutor`, and `BaseVLMSARMMonitorBackend`. The remaining
sim-to-real blockers are:

- replace `SimulatorSignalVLMSARMMonitor` with the real VLM/SARM backend;
- add a `RealWorldCRIEEnvAdapter` that exposes `reset`, `get_obs`,
  `describe_obs`, and `get_reward_done` for real tasks;
- wrap the learned policy executor in `rocobench/skills/learned` as a CRIE-BT
  `BaseSkillExecutor`;
- add guarded primitive execution, safety checks, operator stop handling, and
  artifact logging before real-world autonomous execution.
