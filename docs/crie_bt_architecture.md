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

Execution / recovery modes come from CRIE-BT:

- `open_loop`: plan once, execute once, no runtime feedback or replanning.
- `direct_feedback`: send executor failure/progress feedback directly to the
  high-level planner for replanning.
- `bt_mediated`: route progress, uncertainty, and failure signals through a
  behavior-tree runtime controller before deciding whether to continue, retry
  locally, explain, request human input, replan, or abort.

The full intended matrix is:

```text
plan   + open_loop
plan   + direct_feedback
plan   + bt_mediated
chat   + open_loop
chat   + direct_feedback
chat   + bt_mediated
dialog + open_loop
dialog + direct_feedback
dialog + bt_mediated
```

Current implementation status:

- implemented: `plan + open_loop`, `chat + open_loop`, `dialog + open_loop`
  through `LegacyPromptPlanner`;
- planned: all `direct_feedback` and `bt_mediated` combinations with
  `plan/chat/dialog`;
- still supported for deterministic smoke tests: `legacy_action`, which consumes
  provided `EXECUTE` blocks and is not an LLM planner mode.

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
- `rocobench/crie_bt/controllers.py`: open-loop, direct-feedback, and BT-mediated controllers.
- `rocobench/crie_bt/communication.py`: research speech/overlay event abstraction.
- `rocobench/crie_bt/failure_injection.py`: scripted failure scenarios.

## Runtime Flow

```text
Planner -> CollaborativePlan -> controller mode
    open_loop: execute each skill once
    direct_feedback: execute, then send failures to planner
    bt_mediated: execute through BT runtime policy
Executor -> ProgressMonitor -> UncertaintyEstimator -> FailureDetector
BT policy -> continue / retry / explain / human input / replan / abort
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

The first implementation is fully testable with scripted planners and
executors. The simulator path now includes existing RoCo RRT execution and
open-loop integration for `plan/chat/dialog`. The next implementation step is
to let `direct_feedback` and `bt_mediated` call those same prompters during
replanning while preserving prompt history and execution feedback.
