# CRIE-BT Architecture

CRIE-BT is an optional architecture mode for behavior-tree-grounded
mixed-initiative multi-agent human-robot collaboration. It is additive: the
legacy RoCo `action_only` and `action_and_path` flows remain unchanged.

## Motivation

The goal is to compare three planner-executor strategies under a common schema:

- `open_loop`: plan once, execute once, no runtime feedback or replanning.
- `direct_feedback`: send executor failure/progress feedback directly to the
  high-level planner for replanning.
- `bt_mediated`: route progress, uncertainty, and failure signals through a
  behavior-tree runtime controller before deciding whether to continue, retry
  locally, explain, request human input, replan, or abort.

## Modules

- `rocobench/crie_bt/types.py`: JSON-serializable dataclasses.
- `rocobench/crie_bt/status.py`: execution, BT, decision, progress, and failure enums.
- `rocobench/crie_bt/planner.py`: scripted planner and LLM adapter interface.
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
executors. RRT and learned executors are stable adapter interfaces with explicit
TODOs for injection of existing RoCo RRT and ACT/Diffusion/LeRobot policy
backends.
