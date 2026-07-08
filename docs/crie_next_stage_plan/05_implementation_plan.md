# Implementation Plan

## Current repo anchors

The uploaded repo already contains useful pieces:

- CRIE-BT modules under `rocobench/crie_bt/`.
- Existing progress helper under `rocobench/crie_bt/progress.py`.
- Existing simulator signal monitor under `rocobench/crie_bt/vlm_sarm_monitor.py`, but this name should be corrected because the simulator-coded monitor belongs to CRIE-BT, not the baseline.
- Existing controllers under `rocobench/crie_bt/controllers.py`.
- Existing RRT and RoCoBench integration under `rocobench/rrt.py`, `rocobench/rrt_multi_arm.py`, and `rocobench/skills/`.
- Existing evaluation scripts under `scripts/run_crie_bt_sim.py`, `scripts/run_crie_bt_eval.py`, and `scripts/eval_sandwich_open_loop.py`.

## Immediate naming correction

Rename or wrap the existing simulator monitor:

Current confusing name:

```text
rocobench/crie_bt/vlm_sarm_monitor.py
SimulatorSignalVLMSARMMonitor
```

Recommended new name:

```text
rocobench/crie_bt/monitors.py
CodedSimProgressMonitor
```

Keep backward-compatible imports temporarily if tests depend on the old name.

## Milestone 0: condition registry and logging cleanup

Goal: all runs should know exactly which condition they are executing.

Tasks:

1. Add `configs/conditions/condition_registry.yaml` or equivalent.
2. Add fields to every episode row:
   - `condition_name`
   - `controller_family`
   - `team_type`
   - `coordination_mode`
   - `skill_backend`
   - `monitor_backend`
   - `monitor_privileged`
   - `environment`
   - `planner_input_type`
3. Add a central function:

```python
def build_condition(condition_name: str, stage: str) -> ConditionConfig:
    ...
```

4. Make `run_crie_bt_sim.py` and future scripts use the condition registry rather than ad hoc `mode` and `planner_mode` combinations.

Acceptance check:

```bash
python scripts/run_condition_matrix.py --stage step1 --dry-run --episodes 1
```

should create rows containing the full condition metadata.

## Milestone 1: Step 1 robot-robot simulation

Goal: evaluate RR conditions using RRT and coded simulator monitor.

Conditions:

- `VLM-RR-Cent`
- `VLM-RR-Dialog`
- `CRIE-BT-RR-Cent`
- `CRIE-BT-RR-Dialog`

Tasks:

- existing RoCoBench tasks first: `sandwich`, `pack`, `cabinet`, `sort`;
- later map/extend to collaborative cooking and medication dispensing if simulator support is available.

Implementation tasks:

1. Build `VLMCentralizedController` for baseline centralized.
2. Build `VLMDialogController` for baseline dialog/distributed.
3. Ensure baseline uses image/status/history feedback but no coded progress monitor.
4. Build `CRIEBTCentralizedController` and `CRIEBTDialogController` with `CodedSimProgressMonitor`.
5. Add RRT skill backend to both baseline and ours.
6. Add `planner_safe_observation()` so baseline planner cannot accidentally consume oracle simulator state.
7. Compute monitor-specific metrics only for CRIE-BT in this stage; baseline does not have a separate monitor.
8. Compute false replan/unnecessary replan using ground-truth progress labels from the simulator/evaluator.

Suggested command:

```bash
python scripts/run_condition_matrix.py \
  --stage step1 \
  --tasks sandwich pack cabinet sort \
  --conditions VLM-RR-Cent VLM-RR-Dialog CRIE-BT-RR-Cent CRIE-BT-RR-Dialog \
  --skill-backend RRT \
  --episodes 10 \
  --seeds 0 1 2 \
  --output results/step1_rr_sim/results.jsonl
```

## Milestone 2: Step 2A human-robot simulation with terminal interface

Goal: replace one robot with a human-controlled simulated agent.

Conditions:

- `VLM-HR-Cent`
- `VLM-HR-Dialog`
- `CRIE-BT-HR-Cent`
- `CRIE-BT-HR-Dialog`

Implementation tasks:

1. Add `HumanAgentAdapter` that can control the second simulated agent through terminal commands.
2. Add simple keyboard commands:
   - `1`: accept instruction;
   - `2`: reject instruction;
   - `3`: counter-propose;
   - `w/a/s/d`: move simulated human-controlled end-effector or selected agent, if applicable;
   - `p`: pick selected object;
   - `d`: drop/place;
   - `done`: mark human instruction as completed.
3. Add `TerminalCommunicationInterface`.
4. In centralized HR, central planner sends explicit human instruction.
5. In dialog HR, robot planner sends proposal/requests to the human.
6. Log human responses, delays, manual actions, and whether the human followed the intended instruction.

Suggested command:

```bash
python scripts/run_condition_matrix.py \
  --stage step2a \
  --tasks medication_sim cooking_sim \
  --conditions VLM-HR-Cent VLM-HR-Dialog CRIE-BT-HR-Cent CRIE-BT-HR-Dialog \
  --human-interface terminal \
  --skill-backend RRT \
  --episodes 5 \
  --output results/step2_hr_terminal/results.jsonl
```

## Milestone 3: Step 2B communication polish

Goal: keep the same interface but add optional speech wrapper.

Tasks:

1. Add `SpeechCommunicationInterface` wrapper around the same send/read API.
2. Keep terminal interface as fallback.
3. Run a small pilot to measure dialogue burden and failure modes.

Do not block core Step 2 on speech. Terminal is enough for the first implementation.

## Milestone 4: Step 3 real-world task integration

Goal: replace RRT with learned skills and coded monitor with SARM.

Tasks:

1. Define real-world task stages for:
   - collaborative medication dispensing;
   - collaborative cooking.
2. Implement or wrap learned visual skills:
   - `pick(object)`;
   - `place(target)`;
   - task-specific subtasks such as `scoop`, `pour`, `handover`, `open`, `close`, if needed.
3. Implement `SARMProgressMonitor` with minimal output:
   - `stage_id`;
   - `progress_score` from 0 to 1;
   - `is_stage_done` thresholded from progress score;
   - optional confidence/message.
4. Create SARM training/evaluation dataset:
   - images/video frames;
   - stage ID;
   - progress label in `[0,1]`.
5. Add real-world safety wrapper:
   - emergency stop;
   - workspace boundary check;
   - timeout;
   - human override.

Suggested command:

```bash
python scripts/run_condition_matrix.py \
  --stage step3 \
  --tasks medication_real cooking_real \
  --conditions VLM-HR-Cent VLM-HR-Dialog CRIE-BT-HR-Cent CRIE-BT-HR-Dialog \
  --skill-backend LearnedSkill \
  --monitor-backend SARM \
  --human-interface terminal_or_speech \
  --episodes 5 \
  --output results/step3_real_hr/results.jsonl
```

## Milestone 5: paper-ready analysis scripts

Tasks:

1. Add `scripts/analyze_condition_matrix.py`.
2. Generate:
   - success-rate table;
   - completion-time table;
   - replanning and retry table;
   - monitor metric table;
   - dialogue burden table;
   - per-task breakdown;
   - ablation table.
3. Export both `.csv` and LaTeX table snippets.

## Minimum tests to add

1. `test_condition_registry_loads_all_8_conditions`.
2. `test_baseline_has_no_explicit_monitor`.
3. `test_criebt_uses_coded_monitor_in_step1_and_step2`.
4. `test_criebt_uses_sarm_monitor_in_step3`.
5. `test_planner_safe_observation_strips_oracle_state`.
6. `test_terminal_human_interface_returns_accept_reject_counter_done`.
7. `test_episode_log_contains_required_metadata`.
8. `test_monitor_progress_score_is_between_0_and_1`.

## Main implementation risk

The most important risk is accidentally giving the baseline planner privileged simulator state. Prevent this with a hard observation filter and log `planner_input_type=image_status_only` for baseline conditions.
