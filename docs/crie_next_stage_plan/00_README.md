# CRIE-BT / SARM Next-Stage Implementation and Paper Artifacts

This package converts the confirmed research plan into implementation and writing artifacts. It assumes the following locked framing:

- **Baseline:** monolithic VLM planner that self-monitors from image feedback. There is no separate monitor module in the baseline.
- **Ours:** CRIE-BT with an explicit progress-monitor module.
- **Simulation monitor:** coded/oracle simulator and RRT-aware progress monitor. It may use simulator/task signals that reveal current subtask success, failure, or progress, including `done` if that is the best available signal.
- **Real-world monitor:** SARM-based progress monitor. SARM means **Stage-Aware Reward Modeling** and outputs a progress score for the current stage, from 0 to 1.
- **Step 1:** robot-robot simulation with RRT skills and coded simulator monitor for CRIE-BT.
- **Step 2:** human-robot simulation, first through a terminal/keyboard human interface, with RRT robot skills and coded simulator monitor for CRIE-BT.
- **Step 3:** real human-robot evaluation for collaborative medication dispensing and collaborative cooking, with learned visual skills and SARM progress monitor.

Recommended reading order:

1. `01_locked_decisions.md`
2. `02_condition_taxonomy.md`
3. `03_architecture_spec.md`
4. `04_interface_api_spec.md`
5. `05_implementation_plan.md`
6. `06_experiment_matrix.md`
7. `07_metrics_and_logging_spec.md`
8. `08_paper_outline.md`
9. `09_methods_section_draft.md`
10. `10_coding_agent_prompt.md`

Machine-readable files:

- `condition_registry.yaml`
- `experiment_matrix.csv`
- `logging_schema.json`
- `monitor_decision_schema.json`
