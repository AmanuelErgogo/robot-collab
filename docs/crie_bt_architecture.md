# CRIE-BT Architecture

CRIE-BT is an optional architecture mode for behavior-tree-grounded
mixed-initiative multi-agent human-robot collaboration. It is additive: the
legacy RoCo `action_only` and `action_and_path` flows remain unchanged.

## Motivation

CRIE-BT evaluation has two orthogonal axes.

Planner / coordination modes come from the existing RoCoBench prompting stack:

- `plan`: one centralized planner prompt produces one synchronized action plan.
- `chat`: one centralized prompt asks for team discussion plus a final
  synchronized action plan.
- `dialog`: per-agent dialogue prompts run in turn until an agent emits the
  final synchronized action plan.

Execution / recovery modes come from CRIE-BT (open-loop retained only for
legacy/debug use and is not evaluated in the paper):

- `direct_feedback`: send executor failure/progress feedback directly to the
  high-level planner for replanning.
- `bt_mediated`: route progress, uncertainty, and failure signals through a
  behavior-tree runtime controller before deciding whether to continue, retry
  locally, explain, request human input, replan, or abort.
- `vlm_sarm_monitor_planner`: execute one planner action and query the
  VLM/SARM monitor interface for `DONE`, `FAILED`, or `IN_PROGRESS`. The
  simulator backend uses simulator done and executor failed signals.

The full intended matrix is:

```text
plan   + direct_feedback
plan   + bt_mediated
plan   + vlm_sarm_monitor_planner
chat   + direct_feedback
chat   + bt_mediated
chat   + vlm_sarm_monitor_planner
dialog + direct_feedback
dialog + bt_mediated
dialog + vlm_sarm_monitor_planner
```

Current implementation status:

- implemented: `plan/chat/dialog + direct_feedback`, `plan/chat/dialog +
  bt_mediated`, and `plan/chat/dialog + vlm_sarm_monitor_planner` through
  `LegacyPromptPlanner`; (open-loop remains available as a legacy option)
- paper simulator methods: `dialog + bt_mediated` (CRIE-BT-Dialog) and
  `dialog + vlm_sarm_monitor_planner`
  (VLM/SARM-Monitor-Planner-Dialog);
- centralized ablations: `chat + bt_mediated` and
  `chat + vlm_sarm_monitor_planner`;
- still supported for deterministic smoke tests: `legacy_action`, which consumes
  provided `EXECUTE` blocks and is not an LLM planner mode.

For paper-facing simulator baselines, `vlm_sarm_monitor_planner` replaces the
historical `direct_feedback` naming. In the simulator backend it is equivalent
to direct-feedback replanning because the monitor reads simulator done/failure
signals, but it logs explicit monitor decisions and keeps the real VLM/SARM
backend interface.

## Modules

- `rocobench/crie_bt/types.py`: JSON-serializable dataclasses.
- `rocobench/crie_bt/status.py`: execution, BT, decision, progress, and failure enums.
- `rocobench/crie_bt/planner.py`: scripted planner and LLM adapter interface.
- `rocobench/crie_bt/legacy_tasks.py`: adapters for existing RoCoBench
  `EXECUTE` action plans and `plan/chat/dialog` prompters.
- `rocobench/crie_bt/executor.py`: scripted, RRT-adapter, and learned-adapter executors.
- `rocobench/crie_bt/progress.py`: robust manipulation progress monitor.
- `rocobench/crie_bt/uncertainty.py`: neutral, heuristic, ensemble, and metadata uncertainty estimates.
- `rocobench/crie_bt/failure.py`: failure detection rules.
- `rocobench/crie_bt/bt_nodes.py`: lightweight BT nodes.
- `rocobench/crie_bt/bt_controller.py`: behavior-tree runtime policy.
- `rocobench/crie_bt/controllers.py`: open-loop, direct-feedback, BT-mediated,
  and VLM/SARM monitor-planner controllers.
- `rocobench/crie_bt/vlm_sarm_monitor.py`: real/simulator-compatible VLM/SARM
  monitor backend interface and simulator done/failed backend.
- `rocobench/crie_bt/communication.py`: research speech/overlay event abstraction.
- `rocobench/crie_bt/failure_injection.py`: scripted failure scenarios.

## Runtime Flow

```text
Planner -> CollaborativePlan -> controller mode
    direct_feedback: execute, then send failures to planner
    bt_mediated: execute through BT runtime policy
    vlm_sarm_monitor_planner: monitor done/failed, then replan on monitor failure
Executor -> ProgressMonitor -> UncertaintyEstimator -> FailureDetector
BT policy -> continue / retry / explain / human input / replan / abort
SimulatorSignalVLMSARMMonitor -> DONE / FAILED / IN_PROGRESS
```

For legacy RoCoBench tasks, `LegacyPromptPlanner` wraps the existing prompters:

```text
SingleThreadPrompter(plan/chat) or DialogPrompter(dialog)
  -> raw EXECUTE / NAME / ACTION response
  -> CRIE-BT CollaborativePlan with one LEGACY_ACTION_PLAN step
  -> LegacyTaskRRTExecutorAdapter
  -> LLMResponseParser
  -> RRTSkillExecutor
```

This preserves the existing planner prompts and parser behavior while allowing
CRIE-BT to supervise execution.

## Behavior Tree Policy

The BT controller uses conservative local recovery:

- success marks the current subtask complete;
- low confidence emits an explanation and continues when progress is ongoing;
- missed grasp, slippage, and no progress trigger bounded local retries;
- repeated failures, occupied targets, wrong-object/target evidence, and safety
  conflicts request planner-level replanning;
- timeout requests replanning by default.

## Integration Boundary

The simulator path includes existing RoCo RRT execution and `plan/chat/dialog`
integration for open-loop, direct-feedback, BT-mediated, and VLM/SARM
monitor-planner execution. The VLM/SARM simulator baseline already exposes the
backend interface expected by a real monitor; the remaining transfer work is to
replace the simulator done/failed backend with real VLM/SARM observations, wrap
learned policy execution as a CRIE-BT `BaseSkillExecutor`, and add a real-world
environment adapter.
