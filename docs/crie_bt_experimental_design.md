# CRIE-BT Experimental Design

This document defines the current paper-facing experiment. The method set has
two primary methods and two centralized ablations.

## 1. Simulator Scope

The Robot-Robot simulator experiment runs autonomous robot agents in RoCoBench.
All agents are controlled through the existing `EXECUTE / NAME / ACTION`
grammar and executed by the legacy RRT path unless otherwise noted.

Paper tasks:

| Task ID | Paper name | Class | Notes |
| --- | --- | --- | --- |
| `sandwich` | Sandwich | Sequential-dependent | Strict stacking order. |
| `pack` | Pack Grocery | Parallel-independent | Flexible item order. Use `--adapter legacy` for LLM Robot-Robot runs. |
| `cabinet` | Cabinet | Gated | Cabinet manipulation with task-specific agent set. |
| `sort` | Sort | Parallel-independent | Object-to-zone assignment. |

Extra supported runner tasks, outside the paper scope: `sweep` and `rope`.

## 2. Primary Methods

| Method | Controller | Communication mode | Runner flags |
| --- | --- | --- | --- |
| CRIE-BT-Dialog | `BTMediatedController` | `dialog` | `--mode bt_mediated --planner-mode dialog --adapter legacy` |
| VLM/SARM-Monitor-Planner-Dialog | `VLMSARMMonitorPlannerController` | `dialog` | `--mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy` |

The VLM/SARM simulator baseline uses the same planner and RRT executor path as
CRIE-BT. Its monitor interface is the same surface expected from the real
VLM/SARM backend, but the simulator backend uses `env.get_reward_done()` for
`DONE` and executor failure feedback for `FAILED`.

In simulator-only experiments, this makes the VLM/SARM monitor-planner baseline
the paper-facing replacement for the older `direct_feedback` controller name.
`VLM/SARM-Monitor-Planner-Dialog` is the reported name for the behavior that was
previously closest to `DirectFeedback-Dialog`; the centralized ablation is the
reported name for the behavior previously closest to `DirectFeedback-Cent`.

## 3. Ablations

| Ablation | Controller | Communication mode | Runner flags |
| --- | --- | --- | --- |
| CRIE-BT-Cent | `BTMediatedController` | `chat` | `--mode bt_mediated --planner-mode chat --adapter legacy` |
| VLM/SARM-Monitor-Planner-Cent | `VLMSARMMonitorPlannerController` | `chat` | `--mode vlm_sarm_monitor_planner --planner-mode chat --adapter legacy` |

Open-loop runs are legacy and not part of the current paper methods or main ablations.

## 4. Result Paths

New results use:

```text
results/robot_robot_sim_v1/{task_id}/{method}/episodes.jsonl
results/robot_robot_sim_v1/{task_id}/{method}/prompts/
results/robot_robot_sim_v1/{task_id}/{method}/analysis/
```

Stable method directory names:

```text
crie_bt_dialog
vlm_sarm_dialog
crie_bt_cent
vlm_sarm_cent
```

Legacy result directories such as `results/sandwich_bt_mediated/` and `results/c3_to_c6_runs/` should not be extended with new paper runs.

## 5. Metrics

| Metric | Episode field | Analyzer field |
| --- | --- | --- |
| Task success rate | `sim_success` | `task_success_rate` |
| Controller success rate | `success` | `success_rate` |
| Completion time | `wall_time_s` | `avg_wall_time_s`, `std_wall_time_s` |
| LLM latency | `llm_call_latencies_s` | `avg_llm_latency_s` |
| Token consumption | `llm_prompt_tokens`, `llm_completion_tokens`, `llm_total_tokens` | `avg_llm_prompt_tokens`, `avg_llm_completion_tokens`, `avg_llm_total_tokens` |
| Replanning | `replans` | `avg_replans` |
| Local recovery | `local_retries` | `avg_local_retries` |
| Failure breakdown | `failure_counts` | `failure_counts` |
| Monitor decisions | `events` with `event_type=VLM_SARM_MONITOR` | inspect event payloads |
| Explanations | `explanations` | `explanation_count` |
| Reactivity | `reactivity_s` when annotated | `avg_reactivity_s` |
| Hallucination rate | `hallucination_count`, `hallucination_annotation_count` | `hallucination_rate` |

Reactivity and hallucination rate are supported by the analyzer but are not
automatically produced by the simulator. They require external annotation or a
future event/judge pipeline.

## 6. Failure Codes

Episodes aggregate `failure_counts` using CRIE-BT failure codes:

| Code | Trigger |
| --- | --- |
| `PLANNER_ERROR` | Planner did not produce a parseable executable response. |
| `POSTCONDITION_FAILED` | The action ran but the expected simulator postcondition was not satisfied. |
| `MISSED_GRASP` | The gripper closed without holding the target object. |
| `NO_PROGRESS` | The executor exceeded its progress budget. |
| `WRONG_OBJECT` | The wrong object was manipulated. |
| `WRONG_TARGET` | The object was placed at the wrong target. |
| `SAFETY_CONFLICT` | The planned action conflicts with a safety constraint. |
| `TIMEOUT` | Execution exceeded the configured budget. |
| `UNKNOWN` | Unclassified executor failure. |

## 7. Real-World Transfer Gaps

The simulator path already uses the interfaces needed for a real environment:
`CollaborativePlan`, `SkillCall`, `ExecutionFeedback`, `BaseSkillExecutor`,
and `BaseVLMSARMMonitorBackend`. Remaining blockers:

- replace `SimulatorSignalVLMSARMMonitor` with the real VLM/SARM backend;
- add a `RealWorldCRIEEnvAdapter` with `reset`, `get_obs`, `describe_obs`, and
  `get_reward_done`;
- wrap learned policy execution from `rocobench/skills/learned` as a CRIE-BT
  `BaseSkillExecutor`;
- add guarded real-world primitive execution, stop handling, safety checks, and
  artifact/video logging.

## 8. Verification

Run the stdlib CRIE-BT tests after code changes:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/crie_bt
```

Run MuJoCo and live LLM experiments only inside the Python 3.8 `roco`
environment.
