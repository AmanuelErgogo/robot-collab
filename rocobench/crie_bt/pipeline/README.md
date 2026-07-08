# CRIE-BT / SARM next-stage pipeline

This package is the clean, swappable architecture layer specified in
[`docs/crie_next_stage_plan/`](../../../docs/crie_next_stage_plan). It replaces the ad hoc
`mode` / `planner_mode` wiring of the legacy runner with a condition registry, a
set of small interfaces, and four collaboration controllers that stay identical
from Step 1 (robot-robot sim) through Step 3 (real-world human-robot).

Importing this package pulls in **no** MuJoCo, dm_control, or LLM client. A
deterministic synthetic backend makes the whole matrix runnable and testable;
heavy backends attach behind documented extension points.

## The two method families

| Family | Monitoring | Recovery |
|---|---|---|
| **Baseline** `VLM-*` | none — the monolithic VLM planner self-monitors from image/status/history/dialogue feedback | must re-plan (an expensive planner call) on every failure |
| **Ours** `CRIE-BT-*` | an **explicit** progress monitor (coded simulator monitor in Step 1/2, SARM in Step 3) | cheap local retries first; re-plan only when the monitor reports stuck/failed |

The baseline is never labelled SARM. SARM is the CRIE-BT **real-world** monitor
backend only.

## The eight conditions

`{VLM, CRIE-BT} × {RR, HR} × {Cent, Dialog}`, resolved from
[`configs/conditions/condition_registry.yaml`](../../../configs/conditions/condition_registry.yaml)
via `build_condition(name, stage)`. Backends (skill/monitor/environment) are
reported separately and are stage properties, not part of the condition name.

| Stage | Team | Skill backend | CRIE-BT monitor | Env |
|---|---|---|---|---|
| step1 | RR | RRT | CodedSim (privileged) | sim |
| step2 / step2a | HR (terminal human) | RRT | CodedSim (privileged) | sim |
| step3 | HR | LearnedSkill | SARM (not privileged) | real |

## Fairness rule

Non-oracle planners must never read `ObservationBundle.oracle_state`. Every
planner call goes through `planner_safe_observation(obs)`. Only the coded
simulator monitor and the evaluator may read `oracle_state`.

## Running the matrix

```bash
# Acceptance check: emit fully-populated condition metadata rows, no execution.
python scripts/run_condition_matrix.py --stage step1 --dry-run --episodes 1 \
    --output results/step1/dry.jsonl

# Step 1 robot-robot simulation (synthetic backend).
python scripts/run_condition_matrix.py --stage step1 \
    --conditions VLM-RR-Cent VLM-RR-Dialog CRIE-BT-RR-Cent CRIE-BT-RR-Dialog \
    --tasks sandwich pack cabinet sort --seeds 0 1 2 --episodes 10 \
    --output results/step1_rr_sim/results.jsonl

# Step 2 human-robot simulation (scripted terminal human; use --human-interface
# terminal for a live keyboard: 1=accept 2=reject 3=counter-propose done=complete).
python scripts/run_condition_matrix.py --stage step2 \
    --conditions VLM-HR-Cent CRIE-BT-HR-Cent \
    --tasks medication_sim cooking_sim --human-interface scripted --episodes 5 \
    --output results/step2_hr_terminal/results.jsonl

# Paper-ready tables (CSV + LaTeX).
python scripts/analyze_condition_matrix.py results/step1_rr_sim/results.jsonl \
    --output-dir results/step1_rr_sim/analysis
```

Each episode row carries the full condition metadata required by
[`docs/crie_next_stage_plan/logging_schema.json`](../../../docs/crie_next_stage_plan/logging_schema.json);
per-episode `events.jsonl` files capture planner calls, skill feedback, monitor
updates, replans, and human interaction.

## Module map

| Module | Responsibility |
|---|---|
| `interfaces.py` | data types + abstract interfaces + `planner_safe_observation` |
| `conditions.py` | `ConditionConfig`, `ConditionRegistry`, `build_condition` |
| `monitors.py` | `CodedSimProgressMonitor`, `SARMProgressMonitor`, `NoSeparateMonitor` |
| `planners.py` | `ScriptedStagePlanner` (oracle-free), `LLMPlannerAdapter` (extension point) |
| `executors.py` | `SyntheticSkillExecutor`, `RRTSkillExecutorAdapter`, `LearnedSkillExecutorAdapter` |
| `env_adapters.py` | `SyntheticEnvironmentAdapter`, `RoCoBenchEnvironmentAdapter` (extension point) |
| `communication.py` | `TerminalCommunicationInterface`, `ScriptedHumanCommunicationInterface` |
| `controllers.py` | the four `CollaborationController`s + `build_collaboration_controller` |
| `episode_logger.py` | `EpisodeLogger` (episode rows + `events.jsonl`) |

Tests live in [`tests/crie_bt/pipeline/`](../../../tests/crie_bt/pipeline) and run
without MuJoCo:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/crie_bt/pipeline -q
```

## Real RoCoBench backend (Step 1)

`roco_backend.py` bridges the pipeline to the real MuJoCo simulator + RRT skill
stack. It is kept out of `pipeline/__init__.py` because importing it *does* pull
in MuJoCo/dm_control. Use an offscreen GL backend:

```bash
MUJOCO_GL=egl python scripts/validate_roco_step1.py --task pack
```

`build_roco_step1(task, seed, responses=...)` returns
`(env_adapter, executor, planner_factory)` wired to `LegacyTaskRRTExecutorAdapter`.
A pipeline `SkillCall` carries a raw RoCoBench `EXECUTE` block in `args["response"]`;
`RoCoScriptedPlanner` supplies fixed `EXECUTE` blocks for a no-LLM plumbing check.

**A real-LLM Robot-Robot experiment already exists and has been run** (Gemini via
Vertex AI) through the legacy runner — do not rebuild it. See
[`docs/crie-bt/running_the_experiment.md`](../../../docs/crie-bt/running_the_experiment.md),
[`docs/crie-bt/crie_bt_conditions_tasks_metrics.md`](../../../docs/crie-bt/crie_bt_conditions_tasks_metrics.md),
and [`scripts/run_crie_bt_sim.py`](../../../scripts/run_crie_bt_sim.py)
(`--mode bt_mediated --planner-mode dialog --llm-source gemini-2.5-flash`). The
open work for this pipeline layer is to have `LLMPlannerAdapter` **wrap that
existing prompter wiring** (`_build_legacy_prompt_planner`) rather than reinvent it.

Environment note: in this checkout the MuJoCo-capable interpreter is
`.venv-google-genai` (no `pytest`), while the `pytest` interpreter
(`miniconda`) lacks `transforms3d` for `rocobench.envs`. The real-sim
`pytest` test is therefore opt-in (`ROCO_SIM_TEST=1`) and auto-skips; validate
the real path with `scripts/validate_roco_step1.py` under the MuJoCo interpreter.

## Status vs. the plan

**Done (runnable + tested on the synthetic backend)**

- Milestone 0 — condition registry, `build_condition`, full episode/event logging.
- Milestone 1 — the four Step 1 robot-robot controllers, coded simulator monitor,
  fairness gate, false/necessary replan vs. local-retry accounting.
- Milestone 2 (Step 2A) — human-robot conditions, terminal + scripted human
  interfaces, human accept/reject/counter/done handling and metrics.
- Milestone 5 — `analyze_condition_matrix.py` (CSV + LaTeX tables).

**Done (validated on the real RoCoBench simulator)**

- `RoCoBenchEnvironmentAdapter` + `RoCoRRTSkillExecutorAdapter` run the new
  controllers over a real MuJoCo task through the legacy RRT stack, with the
  coded monitor reading real `get_reward_done` oracle state. Validated end to end
  via `scripts/validate_roco_step1.py` (a WAIT action executes through RRT and is
  logged with `environment=sim`, `monitor_backend=CodedSim`). Two Pydantic V1→V2
  breaks in `rocobench/envs/{env_utils,base_env}.py` were fixed to unblock this.

**Done (merged with the legacy real-LLM runner)** — see
[`docs/crie-bt/crie_bt_pipeline_merge.md`](../../../docs/crie-bt/crie_bt_pipeline_merge.md)

- `LLMPlannerAdapter` bridges any legacy `BasePlanner` (incl. the real-LLM
  `LegacyPromptPlanner`) into the pipeline. The real Gemini/OpenAI prompter + RRT
  wiring now lives once in `rocobench/crie_bt/roco_runtime.py`, used by **both**
  `scripts/run_crie_bt_sim.py` and `build_roco_condition`.
- `build_roco_condition(condition, task, ...)` and
  `run_condition_matrix.py --backend roco --llm-source gemini-2.5-flash` run the
  real path through the registry with full condition metadata. Bridge mechanics
  are unit-tested with a mock planner; a full real-LLM run needs credentials
  (`docs/getting-started/llm_setup_and_credentials.md`) and the MuJoCo interpreter.

**Remaining**

- Run the four Step-1 conditions on `--backend roco` with real credentials and
  compare against the legacy `run_crie_bt_sim.py` numbers (equivalence check).
- Milestone 3 — `SpeechCommunicationInterface` wrapper (terminal stays the fallback).
- Milestone 4 (Step 3) — learned visual skills, a trained `SARMProgressMonitor`
  scorer + its stage/progress dataset, real-world safety wrapper, hardware.
