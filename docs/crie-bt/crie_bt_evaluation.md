# CRIE-BT Evaluation

CRIE-BT includes a stdlib-only scripted evaluation path for controller smoke
tests and a RoCoBench simulator path for paper runs.

> **Canonical runner (decided 2026-07-08):** run paper evaluations through the
> pipeline — [running_the_experiment.md](running_the_experiment.md) and
> [run_single_episode.md](run_single_episode.md). The `--mode` / `--planner-mode`
> flags and `VLM/SARM-Monitor-Planner` naming below are the **deprecated legacy
> runner** (`scripts/run_crie_bt_sim.py`), kept for reproducing prior runs. The
> pipeline equivalents are the condition names `CRIE-BT-RR-{Cent,Dialog}` /
> `VLM-RR-{Cent,Dialog}` (mapping in
> [crie_bt_pipeline_merge.md](crie_bt_pipeline_merge.md)).

## Legacy paper methods (deprecated)

| Method | Legacy runner flags | Pipeline condition |
|---|---|---|
| CRIE-BT-Dialog | `--mode bt_mediated --planner-mode dialog --adapter legacy` | `CRIE-BT-RR-Dialog` |
| VLM/SARM-Monitor-Planner-Dialog | `--mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy` | `VLM-RR-Dialog` |
| CRIE-BT-Cent | `--mode bt_mediated --planner-mode chat --adapter legacy` | `CRIE-BT-RR-Cent` |
| VLM/SARM-Monitor-Planner-Cent | `--mode vlm_sarm_monitor_planner --planner-mode chat --adapter legacy` | `VLM-RR-Cent` |

`direct_feedback` remains implemented for debugging and legacy comparisons, but it is not part of the current paper method set.

For paper reporting, the VLM/SARM simulator baseline is the replacement for
the old direct-feedback baseline. The simulator monitor consumes done/failure
signals and triggers replanning through the monitor interface, so it is
behaviorally equivalent to direct-feedback in simulation while producing
`VLM_SARM_MONITOR` event logs and matching the real VLM/SARM backend surface.

## Scripted Smoke Evaluation

This path does not require LLM API keys, learned checkpoints, MuJoCo, or
LeRobot.

```bash
python scripts/run_crie_bt_eval.py \
  --task pack \
  --mode vlm_sarm_monitor_planner \
  --episodes 2 \
  --executor scripted \
  --planner scripted \
  --output results/crie_bt/vlm_sarm_scripted.jsonl
```

Supported scripted modes:

```text
direct_feedback
bt_mediated
vlm_sarm_monitor_planner
all
```

`all` preserves the historical smoke-test set: `direct_feedback` and `bt_mediated`.

## Simulator Evaluation

Use the existing RoCoBench simulator tasks through the CRIE-BT runner:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
  --task sandwich \
  --adapter legacy \
  --mode vlm_sarm_monitor_planner \
  --planner-mode dialog \
  --episodes 1 \
  --uncertainty policy_metadata \
  --output results/robot_robot_sim_v1/sandwich/vlm_sarm_dialog/episodes.jsonl \
  --prompt-artifact-dir results/robot_robot_sim_v1/sandwich/vlm_sarm_dialog/prompts
```

Supported simulator task IDs:

```text
pack
sort
sweep
sandwich
rope
cabinet
all
```

The paper task IDs are `sandwich`, `pack`, `cabinet`, and `sort`. Use
`--adapter legacy` for all four Robot-Robot LLM simulator runs.

## VLM/SARM Simulator Baseline

The VLM/SARM baseline is implemented by
`VLMSARMMonitorPlannerController` and `SimulatorSignalVLMSARMMonitor`.

The monitor interface is:

```text
BaseVLMSARMMonitorBackend.evaluate(env, observation, feedback, step_index, max_steps)
  -> VLMSARMMonitorDecision(status, should_replan, is_done, is_failed, evidence)
```

In simulator runs:

| Monitor decision | Source |
| --- | --- |
| `DONE` | `env.get_reward_done(observation)[1]` |
| `FAILED` | executor feedback failure or `BTStatus.FAILURE` |
| `IN_PROGRESS` | no done or failed signal |

This mirrors the intended real `vlm_sarm_real` interface while avoiding VLM
dependencies in the RoCo simulator process.

## Episode Log Schema

Each JSONL row contains the common controller fields:

```text
mode
task
success
steps
completed_subtasks
failed_subtasks
planner_calls
replans
local_retries
failure_counts
events
subtask_results
explanations
```

Simulator rows also add:

```text
episode
task_id
adapter
planner_mode
paper_method
sim_success
wall_time_s
llm_call_latencies_s
llm_prompt_tokens
llm_completion_tokens
llm_total_tokens
initial_scene
final_scene
task_spec
```

The VLM/SARM baseline logs `VLM_SARM_MONITOR` events with
`payload.monitor_decision`.

## Analyze Logs

```bash
python scripts/analyze_crie_bt_eval.py \
  results/robot_robot_sim_v1/sandwich/vlm_sarm_dialog/episodes.jsonl \
  --output-dir results/robot_robot_sim_v1/sandwich/vlm_sarm_dialog/analysis \
  --group-by task_method
```

Aggregated fields include success rate, task success rate, average steps, wall
time, LLM latency, token usage, planner calls, replans, local retries, failure
counts, recovery rate, explanation count, and annotation-backed reactivity and
hallucination metrics when those fields are present.
