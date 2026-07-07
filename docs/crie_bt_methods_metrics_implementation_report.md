# CRIE-BT Methods and Metrics Implementation Report

## Scope

Current paper-ready simulator scope is Robot-Robot execution on four tasks:
`sandwich`, `pack`, `cabinet`, and `sort`. New results should be written to:

```text
results/robot_robot_sim_v1/{task_id}/{method}/episodes.jsonl
```

Prompt artifacts should go under the matching `prompts/` directory, and analyzer
outputs should go under the matching `analysis/` directory.

## Primary Methods

Only two methods are primary paper methods:

| Method | Status | How it is implemented |
| --- | --- | --- |
| CRIE-BT-Dialog | Implemented for simulator | `scripts/run_crie_bt_sim.py --mode bt_mediated --planner-mode dialog --adapter legacy`. The per-agent `DialogPrompter` is wrapped by `LegacyPromptPlanner`, actions execute through `LegacyTaskRRTExecutorAdapter`, and recovery is mediated by `BTMediatedController`. |
| VLM/SARM-Monitor-Planner-Dialog | Implemented simulator baseline | `scripts/run_crie_bt_sim.py --mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy`. The planner and executor path matches CRIE-BT, but recovery decisions come from a VLM/SARM monitor interface. In simulation, `SimulatorSignalVLMSARMMonitor` fills that interface from simulator `done` and executor failure signals instead of real visual/SARM predictions. |

## Ablations

| Ablation | Status | How it is implemented |
| --- | --- | --- |
| CRIE-BT-Cent | Implemented | `--mode bt_mediated --planner-mode chat --adapter legacy`. Centralized planner with BT-mediated recovery. |
| VLM/SARM-Monitor-Planner-Cent | Implemented | `--mode vlm_sarm_monitor_planner --planner-mode chat --adapter legacy`. Centralized planner with the simulator-backed VLM/SARM monitor interface. |
| OpenLoop-Dialog | Implemented smoke/ablation path | `--mode open_loop --planner-mode dialog --adapter legacy`. One planner call with no recovery. |
| OpenLoop-Cent | Implemented smoke/ablation path | `--mode open_loop --planner-mode chat --adapter legacy`. One centralized planner call with no recovery. |

`direct_feedback` remains available in code for debugging and historical
comparisons, but it is not part of the current paper method set.

For simulator paper reporting, the VLM/SARM monitor-planner simulator baseline
replaces the old direct-feedback naming. With `SimulatorSignalVLMSARMMonitor`,
`VLM/SARM-Monitor-Planner-Cent` is behaviorally equivalent to
`DirectFeedback-Cent`, and `VLM/SARM-Monitor-Planner-Dialog` is behaviorally
equivalent to `DirectFeedback-Dialog`, except that the VLM/SARM version logs
explicit `VLM_SARM_MONITOR` decisions and exposes the same backend interface
used by the real VLM/SARM monitor.

## VLM/SARM Simulator Interface

The simulator baseline uses `rocobench/crie_bt/vlm_sarm_monitor.py`:

| Interface | Purpose |
| --- | --- |
| `BaseVLMSARMMonitorBackend.evaluate(...)` | Stable monitor call that a real VLM/SARM backend can implement. |
| `VLMSARMMonitorDecision` | JSON-serializable monitor output with `DONE`, `FAILED`, or `IN_PROGRESS`, plus evidence and `should_replan`. |
| `SimulatorSignalVLMSARMMonitor` | Simulator backend that maps `env.get_reward_done()` to `DONE` and executor failure feedback to `FAILED`. |

This keeps the baseline swappable with a real `vlm_sarm_real` backend while
avoiding VLM dependencies in the simulator path.

## Metrics

| Metric | Status | Implementation |
| --- | --- | --- |
| Task success rate | Implemented | `run_crie_bt_sim.py` records `sim_success` from `env.get_reward_done()`. `analyze_crie_bt_eval.py` reports `task_success_rate`. |
| Controller success rate | Implemented | Controllers record `success`; analyzer reports `success_rate`. |
| Completion time | Implemented | `run_crie_bt_sim.py` records `wall_time_s` around each controller episode; analyzer reports `avg_wall_time_s` and `std_wall_time_s`. |
| Token consumption | Implemented for live LLM runs | The runner wraps each prompter's `query_once`, records returned `usage`, and stores `llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens`; analyzer reports averages. |
| LLM latency | Implemented | The same wrapper records `llm_call_latencies_s`; analyzer reports `avg_llm_latency_s`. |
| Recovery behavior | Implemented | Controllers record `planner_calls`, `replans`, `local_retries`, `failure_counts`, `events`, and `explanations`; analyzer aggregates them. |
| Monitor decisions | Implemented for VLM/SARM baseline | `VLMSARMMonitorPlannerController` logs `VLM_SARM_MONITOR` events with each monitor decision. |
| Reactivity | Analyzer-supported, annotation required | Analyzer reads `reactivity_s` when present. The simulator does not yet automatically infer failure-to-correction latency from events. |
| Hallucination rate | Analyzer-supported, annotation required | Analyzer reads `hallucination_count` and `hallucination_annotation_count` when present. No automatic hallucination judge is implemented. |

## Simulator-To-Real Blockers

- Replace `SimulatorSignalVLMSARMMonitor` with the real VLM/SARM backend while
  preserving the `BaseVLMSARMMonitorBackend.evaluate(...)` interface.
- Learned policy execution exists under `rocobench/skills/learned`, but it is
  not yet wrapped as a CRIE-BT `BaseSkillExecutor`.
- A real-world CRIE-BT environment adapter is still needed. It must expose the
  same episode surface used by the simulator runner: `reset`, `get_obs`,
  `describe_obs`, and `get_reward_done`.
- Real-world execution needs guarded primitive calls, safety stops, operator
  intervention logging, and artifact/video logging before autonomous runs.

## Verification Commands

Use these checks after code changes:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/crie_bt

python scripts/analyze_crie_bt_eval.py \
  results/robot_robot_sim_v1/all_primary.jsonl \
  --output-dir results/robot_robot_sim_v1/analysis \
  --group-by task_method
```

Run MuJoCo and live LLM experiments only inside the `roco` environment.
