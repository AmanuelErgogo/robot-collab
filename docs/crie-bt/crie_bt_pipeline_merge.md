# Merging the legacy runner and the pipeline layer

This document explains how the two Robot-Robot code paths are being merged into
one system that follows `docs/crie_next_stage_plan/`, based on a line-by-line reading
of both.

## Decision (2026-07-08): the pipeline is canonical

The equivalence check found the two paths behaviourally equivalent (exact match on
the deterministic `pack` case; agreement on `success` / `steps` / `planner_calls`
on `sandwich`). The only differences are **definitional** (`replans` is counted
differently — compare `planner_calls`) or **reconciled** (the pipeline's coded
monitor now applies the same postcondition/timeout check as the legacy
`FailureDetector`).

**Going forward, the pipeline (`run_condition_matrix.py --backend roco`, and the
`rocobench/crie_bt/pipeline/` layer) is the main path.** The legacy
`scripts/run_crie_bt_sim.py` is **deprecated but kept for now** as a reference /
fallback; it will be removed in a future cleanup once the pipeline has produced the
paper runs. Both still share one engine via `rocobench/crie_bt/roco_runtime.py`,
so there is no duplicated LLM/RRT wiring in the meantime.

Before deleting legacy: re-run a couple of seeds through both with the postcondition
reconciliation in place and confirm `success` (and now `failed_subtasks`) agree.

## The two paths (before merge)

| | `scripts/run_crie_bt_sim.py` (legacy) | `rocobench/crie_bt/pipeline/` (new) |
|---|---|---|
| Selects a run by | `--mode` + `--planner-mode` | condition registry (8 names) |
| LLM planner | **real** Gemini/GPT via `prompting` prompters | scripted / synthetic |
| Skills + env | **real** RRT + MuJoCo | synthetic (+ `roco_backend` WAIT) |
| Monitor | `SimulatorSignalVLMSARMMonitor` | `CodedSim` / `SARM` / `NoSeparate` |
| Naming | `CRIE-BT-Dialog`, `VLM/SARM-Monitor-Planner` | `VLM-*` / `CRIE-BT-*` |
| Fairness gate | none (relies on `describe_obs` text) | `planner_safe_observation()` |
| Human-robot | no | yes (terminal) |
| Logging | `mode`, `paper_method`, `sim_success` | full condition metadata |

`docs/crie_next_stage_plan/05_implementation_plan.md` explicitly says to make the
runner "use the condition registry rather than ad hoc `mode` and `planner_mode`
combinations," so the target is: **the pipeline layer is canonical; the legacy
real LLM + RRT machinery becomes the backend behind it.**

## Line-by-line pivots that make the merge clean

1. **Legacy planners share one interface.** `LegacyPromptPlanner` (real LLM),
   `LegacyActionPlanner` (fixed EXECUTE blocks) and `ScriptedPlanner` all subclass
   `rocobench.crie_bt.planner.BasePlanner`:
   `generate_plan(goal, obs, feedback, context) -> CollaborativePlan`,
   `notify_result(success)`, `reset_episode()`.
   → **One bridge** (`LLMPlannerAdapter`) converts any of them into the pipeline
   `PlannerInterface`.

2. **Plans already carry EXECUTE blocks.** `LegacyPromptPlanner.generate_plan`
   returns steps whose `SkillCall.arguments["response"]` is a raw RoCoBench
   EXECUTE block (`legacy_tasks.py:386`). `LegacyTaskRRTExecutorAdapter` executes
   exactly that (`legacy_tasks.py:534`). The pipeline
   `RoCoRRTSkillExecutorAdapter` already forwards `SkillCall.args["response"]` to
   that legacy executor. → planner output → executor input needs **no new glue**.

3. **Coordination maps to prompter type.** Legacy `--planner-mode chat` uses
   `SingleThreadPrompter` (centralized); `dialog` uses `DialogPrompter`
   (multi-agent negotiation inside one call). The pipeline coordination axis
   (`Cent`/`Dialog`) therefore selects the prompter; the real per-agent dialogue
   lives inside the prompter (and its saved prompt artifacts), while the pipeline
   controller records `coordination_mode` and coarse dialogue events.

4. **Monitor renames.** The legacy baseline monitor
   `SimulatorSignalVLMSARMMonitor` is the coded simulator monitor; in the pipeline
   it is `CodedSimProgressMonitor`. The baseline is **not** labelled SARM.

## Condition → legacy backend mapping

| Pipeline condition | family / coord | prompter (`planner_mode`) | monitor |
|---|---|---|---|
| `VLM-RR-Cent` | VLM / Cent | `SingleThreadPrompter` (`chat`) | none (self-monitor) |
| `VLM-RR-Dialog` | VLM / Dialog | `DialogPrompter` (`dialog`) | none (self-monitor) |
| `CRIE-BT-RR-Cent` | CRIE-BT / Cent | `SingleThreadPrompter` (`chat`) | `CodedSimProgressMonitor` |
| `CRIE-BT-RR-Dialog` | CRIE-BT / Dialog | `DialogPrompter` (`dialog`) | `CodedSimProgressMonitor` |

The controller family (`VLM` vs `CRIE-BT`) already encodes the old
`vlm_sarm_monitor_planner` vs `bt_mediated` distinction — the pipeline controller
does the monitor/local-retry/replan logic, so the legacy *controllers* are not
needed; only the legacy *planner* and *executor* are reused.

## Implementation

- `LLMPlannerAdapter` (`pipeline/planners.py`) — bridges any legacy `BasePlanner`
  to `PlannerInterface`; calls the legacy planner's `notify_result(success)`
  before each new plan so LLM history stays correct; strips oracle state via the
  fairness gate; reads the raw `EnvState` from the bound env adapter.
- `build_legacy_prompt_planner` / `build_legacy_rrt_executor`
  (`rocobench/crie_bt/roco_runtime.py`) — the prompter/planner/executor
  construction extracted from `run_crie_bt_sim.py` so **both** the legacy runner
  and the pipeline build them from one place (no duplication).
- `build_roco_condition(condition, task, seed, llm_source, ...)`
  (`pipeline/roco_backend.py`) — resolves a condition to a real env + real RRT
  executor + real LLM planner + the right pipeline controller/monitor.
- `run_condition_matrix.py --backend roco` runs the real path through the
  registry with full condition metadata.

## Validation status

- Bridge mechanics validated with a mock prompter (no LLM) and the synthetic
  backend (`tests/crie_bt/pipeline/`).
- Full real-LLM runs require Gemini/OpenAI credentials and the MuJoCo interpreter
  (`.venv-google-genai`, `MUJOCO_GL=egl`); use the same credentials described in
  `docs/getting-started/llm_setup_and_credentials.md` and the commands in
  `docs/crie-bt/running_the_experiment.md`.
