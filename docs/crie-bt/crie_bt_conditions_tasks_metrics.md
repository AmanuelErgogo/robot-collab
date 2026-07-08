# CRIE-BT: Conditions, Tasks, and Metrics

## 1. Experimental Conditions

Each condition is a combination of a **method family** (baseline vs CRIE-BT),
a **team type** (Robot-Robot or Human-Robot), and a **communication mode**
(centralized chat or distributed dialog). The current simulator evaluation
focuses on Robot-Robot teams and implements these choices through runner
execution modes.

### 1.1 Method Families

| Family | Description |
|---|---|
| Baseline | The VLM directly handles task decomposition, subtask allocation, and execution-level control. After each subtask, it decides whether to execute the next subtask or replan when execution fails or when another agent perturbs the plan, e.g. a human or robot starts a different subtask. |
| CRIE-BT / Ours | The VLM decomposes the task, allocates a subtask once, and emits that subtask for Behavior Tree execution. The Behavior Tree plus progress monitor then handles execution monitoring, local recovery, and replanning triggers. |

### 1.2 Execution Modes

| Mode | Class | Description |
|---|---|---|
<!-- `open_loop` removed from evaluated paper methods: open-loop (single-call, no replanning) is retained in codebase for legacy/debug but is not part of paper evaluations. -->
| `direct_feedback` | `DirectFeedbackController` | Legacy reactive baseline/debug mode. The LLM plans one step, the executor runs it, and failures are fed directly back to the LLM for replanning until the task succeeds or `max_steps` is reached. |
| `bt_mediated` | `BTMediatedController` | CRIE-BT implementation. The VLM proposes the next subtask, while a Behavior Tree runtime and progress monitor decide whether to continue, retry locally, or request replanning. |
| `vlm_sarm_monitor_planner` | `VLMSARMMonitorPlannerController` | Paper-facing baseline implementation. The VLM planner controls subtask selection and replanning, while a VLM/SARM-style monitor reports `DONE`, `FAILED`, or `IN_PROGRESS`; in simulation this monitor maps simulator done/failed signals into the same interface. |

**Paper scope**: the two primary methods are the Dialog variants of
`bt_mediated` and `vlm_sarm_monitor_planner`; the Centralised variants are
ablations.

### 1.3 Communication Modes

These determine how the LLM is queried inside any execution mode.

| Mode | Prompter | Description |
|---|---|---|
| `plan` | `SingleThreadPrompter` (no history) | A **single centralised prompt** sent once. No message history. The LLM sees the current observation and task goal and returns a joint action plan for both robots. Cheapest, most deterministic. |
| `chat` | `SingleThreadPrompter` (with history) | Centralised prompt that carries the **full conversation history** from prior replans. The LLM can reason about what was tried before. Slower due to growing prompt size. |
| `dialog` | `DialogPrompter` | **Multi-turn per-agent negotiation**. Chad and Dave alternate sending messages; each sees the other's last message. Produces richer coordination but is the most expensive and prone to parse failures (one agent may submit a single-agent block). |

### 1.4 Condition Matrix

|  | `plan` | `chat` | `dialog` |
|---|---|---|---|
<!-- `open_loop` row removed: open-loop is not evaluated in the paper condition matrix -->
| `direct_feedback` | implemented | implemented | implemented |
| `bt_mediated` | implemented | implemented | implemented |
| `vlm_sarm_monitor_planner` | implemented | implemented | implemented |

The current paper methods are `bt_mediated + dialog` (CRIE-BT-Dialog) and
`vlm_sarm_monitor_planner + dialog`
(VLM/SARM-Monitor-Planner-Dialog). The `chat` versions are centralized
ablations. Use `--adapter legacy` for all four paper simulator tasks in
Robot-Robot runs.

For paper naming, `vlm_sarm_monitor_planner` replaces the older
`direct_feedback` baseline. In simulation the VLM/SARM monitor consumes the
same done/failure signals that direct feedback uses, so the behavior is
equivalent apart from explicit monitor-decision logging and the swappable
real-monitor interface.

---

## 2. Tasks

All tasks run in **MuJoCo** through RoCoBench task definitions. Sandwich and
Pack Grocery use two robot arms in the common setup; Cabinet and Sort may
expose more than two task agents. High-level actions are parsed through each
task's existing `EXECUTE / NAME / ACTION` grammar.

### 2.1 Sandwich (primary benchmark)

**File**: `rocobench/envs/task_sandwich.py`

Assemble a bacon sandwich by stacking items in the correct order on a plate. The recipe is fixed:

```
bread_slice1 → bacon → cheese → tomato → bread_slice2
```

- ~10 sequential pick-and-place actions to complete
- Requires strict ordering; wrong order = failure
- At random initialisation the food items (bacon, cheese, tomato) are shuffled on the table
- `get_reward_done()` returns `done=True` when each item in the recipe is correctly stacked on the previous one

### 2.2 Pack Grocery

**File**: `rocobench/envs/task_pack.py`

Pack a set of grocery items from the table into a box. Order is flexible but both robots must cooperate to avoid collision.

### 2.3 Additional Paper Tasks

| Task | File | Description |
|---|---|---|
| Cabinet | `task_cabinet.py` | Open cabinet door, place objects inside |
| Sort | `task_sort.py` | Sort objects by category into separate zones |

### 2.4 Extra Supported Environments

The runner also supports these RoCoBench tasks, but they are outside the
four-task paper scope:

| Task | File | Description |
|---|---|---|
| Sweep | `task_sweep.py` | Cooperative sweeping of scattered objects |
| Rope | `task_rope.py` | Bimanual rope manipulation |

---

## 3. Metrics

### 3.1 Primary Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Controller Success Rate** | `success_rate` | Episodes where `success=True`. |
| **Task Completion Rate (TCR)** | `task_success_rate` | Episodes where `sim_success=True`; `env.get_reward_done()` returned done for the full task. |
| **95% CI** | Wilson score interval when computed | Tighter than normal approximation for small n. The generic analyzer reports rates. |

<!-- Note: legacy open-loop ASR analyses are not part of the current paper methods. -->

### 3.2 Efficiency Metrics

| Metric | Key | Description |
|---|---|---|
| Steps | `avg_steps` | Total executor `.step()` calls per episode (one call = one simulator tick) |
| Wall time | `avg_wall_time_s ± std` | Total episode wall-clock time in seconds, recorded as `wall_time_s` by `run_crie_bt_sim.py` |
| LLM latency | `avg_llm_latency_s` | Mean per-call LLM response time from `llm_call_latencies_s` |
| Token consumption | `avg_llm_prompt_tokens`, `avg_llm_completion_tokens`, `avg_llm_total_tokens` | Aggregated from each prompter call's LLM usage payload |
| Planner errors | `planner_errors` (count/n) | Episodes that exhausted all `num_replans` retries without a valid EXECUTE block |
| Replans | `avg_replans` | Mean number of LLM replan attempts per episode |
| Local retries | `avg_local_retries` | Mean executor-level retries (sub-LLM recovery) |
| Monitor decisions | `VLM_SARM_MONITOR` events | VLM/SARM baseline decisions and evidence |

Annotation-only paper metrics:

| Metric | Key | Description |
|---|---|---|
| Reactivity | `reactivity_s` | Latency from failure/state change to first corrective action; supported by the analyzer when externally annotated. |
| Hallucination rate | `hallucination_count / hallucination_annotation_count` | Supported by the analyzer for manual or external annotations. |

### 3.3 Failure Codes

Every episode logs a `failure_counts` dict keyed by `FailureCode`:

| Code | Trigger |
|---|---|
| `PLANNER_ERROR` | LLM never returned a parseable EXECUTE block within `num_replans` |
| `POSTCONDITION_FAILED` | RRT ran but the skill's postcondition was not satisfied (e.g. object not grasped) |
| `MISSED_GRASP` | Gripper closed but object not detected as held |
| `NO_PROGRESS` | Executor exceeded sim step budget without completing |
| `UNKNOWN` | Catch-all for unclassified executor exceptions |

---

## 4. Legacy Results Summary

| Mode | ASR (%) | 95% CI | Steps | Time (s) | LLM Lat (s) | P.Err |
|---|---|---|---|---|---|---|
| Plan (Centralised) | 100.0 | [80, 100] | 1.0 ± 0.0 | 9.3 ± 0.3 | 4.9 | 0/15 |
| Chat (w/ History) | 100.0 | [80, 100] | 1.0 ± 0.0 | 15.6 ± 1.6 | 12.5 | 0/15 |
| Dialog (Multi-agent) | 86.7 | [62, 96] | 0.9 ± 0.4 | 18.5 ± 11.7 | 16.0 | 2/15 |

Raw data and LaTeX tables for legacy experiments have been archived and are available on request.

New paper runs should use `results/robot_robot_sim_v1/{task_id}/{method}/`.
