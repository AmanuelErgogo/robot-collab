# Documentation index

Docs are grouped by **topic** so a reader can find things by subject. Start here,
then open the folder that matches what you need.

| Folder | Read it when you want to… |
|---|---|
| [getting-started/](getting-started/) | install, set up credentials, or build an environment for the first time |
| [crie-bt/](crie-bt/) | understand, run, or evaluate the CRIE-BT method (the current system) |
| [skills/](skills/) | work with the learned low-level skills (data → train → infer → execute) |
| [build-phases/](build-phases/) | read the historical phase-by-phase build log (archive) |
| [paper/](paper/) | paper artifacts: environment description, benchmark, writing prompts |
| [crie_next_stage_plan/](crie_next_stage_plan/) | the locked next-stage design spec (conditions, interfaces, milestones) the `pipeline/` layer implements |

## getting-started/

- [llm_setup_and_credentials.md](getting-started/llm_setup_and_credentials.md) — **LLM/Gemini/OpenAI credentials** and env vars for real runs (start here for LLM setup).
- [environment_construction_tutorial.md](getting-started/environment_construction_tutorial.md) — build a RoCoBench MuJoCo environment from scratch.

## crie-bt/ — the current method

- [running_the_experiment.md](crie-bt/running_the_experiment.md) — **runbook** for the Robot-Robot simulator experiment (real Gemini LLM + RRT).
- [run_single_episode.md](crie-bt/run_single_episode.md) — run one task with a chosen method, record video, watch the planner output.
- [crie_bt_conditions_tasks_metrics.md](crie-bt/crie_bt_conditions_tasks_metrics.md) — the conditions, tasks, and metrics (reference).
- [crie_bt_architecture.md](crie-bt/crie_bt_architecture.md) — architecture: planner → behavior tree → progress monitor.
- [crie_bt_evaluation.md](crie-bt/crie_bt_evaluation.md) — how to run and analyze evaluations.
- [equivalence_check.md](crie-bt/equivalence_check.md) — verify the new pipeline matches the legacy runner (same task/seed).
- [crie_bt_api.md](crie-bt/crie_bt_api.md) — module/API reference.
- [crie_bt_experimental_design.md](crie-bt/crie_bt_experimental_design.md) — experimental design and rationale.
- [crie_bt_methods_metrics_implementation_report.md](crie-bt/crie_bt_methods_metrics_implementation_report.md) — methods/metrics implementation report.
- [crie_bt_sim_to_real_plan.md](crie-bt/crie_bt_sim_to_real_plan.md) — simulator → real-robot roadmap.
- [crie_bt_pipeline_merge.md](crie-bt/crie_bt_pipeline_merge.md) — how the new `pipeline/` registry layer merges with the legacy runner.

## skills/ — learned low-level skills

- [learned_subtask_skills.md](skills/learned_subtask_skills.md) — overview of the learned subtask skills.
- [phase0_lerobot_roco_bridge.md](skills/phase0_lerobot_roco_bridge.md) — the LeRobot ↔ RoCo bridge.
- [phase1_skill_architecture.md](skills/phase1_skill_architecture.md) — skill architecture.
- [phase2_dataset_pipeline.md](skills/phase2_dataset_pipeline.md) — expert dataset pipeline.
- [phase3_act_training.md](skills/phase3_act_training.md) / [phase3_short_tutorial.md](skills/phase3_short_tutorial.md) — ACT policy training (+ quick tutorial).
- [phase4_direct_inference.md](skills/phase4_direct_inference.md) — direct ACT inference through the bridge.
- [phase5_learned_executor.md](skills/phase5_learned_executor.md) / [phase5_short_tutorial.md](skills/phase5_short_tutorial.md) — learned executor + deployment boundary (+ quick tutorial).

## build-phases/ — historical build log (archive)

- [phase0_setup_install_test_demos.md](build-phases/phase0_setup_install_test_demos.md), [phase1_setup_install_test_demos.md](build-phases/phase1_setup_install_test_demos.md), [phase2_setup_install_test_demos.md](build-phases/phase2_setup_install_test_demos.md) — per-phase setup/install/test/demo runbooks.
- [phase6_planner_integration.md](build-phases/phase6_planner_integration.md) / [phase6_demo_tutorial.md](build-phases/phase6_demo_tutorial.md) — LLM planner integration (+ demo).
- [phase7_multi_agent_execution.md](build-phases/phase7_multi_agent_execution.md) — multi-agent execution.
- [phase8_benchmark_release.md](build-phases/phase8_benchmark_release.md) / [phase8_demo_tutorial.md](build-phases/phase8_demo_tutorial.md) — benchmark release (+ demo).

## paper/

- [paper_environment_description.md](paper/paper_environment_description.md) — environment description for the paper.
- [crie_bench.md](paper/crie_bench.md) — CRIE-Bench: unified collect/train/test/skills workflow and benchmark.
- [overleaf_paper_prompt.md](paper/overleaf_paper_prompt.md) — Overleaf paper-writing prompt.

## crie_next_stage_plan/ — locked design spec

The confirmed next-stage plan that `rocobench/crie_bt/pipeline/` implements: locked
decisions, condition taxonomy, architecture + interface specs, experiment matrix,
metrics/logging spec, and the machine-readable `condition_registry.yaml` /
`logging_schema.json` / `monitor_decision_schema.json`. Read
[crie_next_stage_plan/00_README.md](crie_next_stage_plan/00_README.md) first for the
recommended reading order. Treat these as the spec of record; the merge doc
[crie-bt/crie_bt_pipeline_merge.md](crie-bt/crie_bt_pipeline_merge.md) tracks how the
code follows it.

---

## Where does a new doc go?

Pick the **first** rule that matches:

1. Setup / install / credentials / first-run → **getting-started/**
2. About the CRIE-BT method — architecture, running it, evaluating it, its conditions/metrics → **crie-bt/**
3. About learned low-level skills — datasets, training, inference, executors → **skills/**
4. A record of a completed build phase (a `phaseN_*` runbook not primarily about skills) → **build-phases/**
5. Paper writing, benchmark spec, or environment description for the paper → **paper/**

Rules of thumb: keep the tree ≤3 levels deep; name files for their subject; when
you add or move a doc, add/adjust its line in this index and fix any links
(`grep -rn 'docs/…\.md'`). Group by **topic** (subject), not by lifecycle.
