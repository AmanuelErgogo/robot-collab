# CRIE-BT: Conditions, Tasks, and Metrics

## 1. Experimental Conditions

Each condition is a combination of an **execution mode** (how the controller reacts to failure) and a **communication mode** (how the LLM is queried).

### 1.1 Execution Modes

| Mode | Class | Description |
|---|---|---|
| `open_loop` | `OpenLoopController` | LLM generates a plan **once** at the start of the episode. All plan steps execute sequentially with no replanning. Failure of one step halts the episode. |
| `direct_feedback` | `DirectFeedbackController` | LLM plans → executes one step → if the step fails, the failure is fed back to the LLM which replans. Loop repeats until the task succeeds or `max_steps` is reached. |
| `bt_mediated` | `BTMediatedController` | Same replanning loop as `direct_feedback` but the decision to replan, retry locally, or request human input is governed by a Behavior Tree runtime that reads uncertainty estimates. |

**Trade-off**: `open_loop` is fastest (one LLM call per episode) but cannot recover from failure. `bt_mediated` is most robust but adds latency and complexity.

### 1.2 Communication Modes

These determine how the LLM is queried inside any execution mode.

| Mode | Prompter | Description |
|---|---|---|
| `plan` | `SingleThreadPrompter` (no history) | A **single centralised prompt** sent once. No message history. The LLM sees the current observation and task goal and returns a joint action plan for both robots. Cheapest, most deterministic. |
| `chat` | `SingleThreadPrompter` (with history) | Centralised prompt that carries the **full conversation history** from prior replans. The LLM can reason about what was tried before. Slower due to growing prompt size. |
| `dialog` | `DialogPrompter` | **Multi-turn per-agent negotiation**. Chad and Dave alternate sending messages; each sees the other's last message. Produces richer coordination but is the most expensive and prone to parse failures (one agent may submit a single-agent block). |

### 1.3 Condition Matrix

|  | `plan` | `chat` | `dialog` |
|---|---|---|---|
| `open_loop` | ✅ evaluated | ✅ evaluated | ✅ evaluated |
| `direct_feedback` | implemented | implemented | implemented |
| `bt_mediated` | implemented | implemented | implemented |

Only the `open_loop` row has been run end-to-end with the real simulator and LLM.

---

## 2. Tasks

All tasks run in **MuJoCo** with two robot arms: **Chad** (`ur5e_robotiq`, right side) and **Dave** (`panda`, left side). Available actions per robot: `PICK <obj>`, `PLACE <target>`, `WAIT`.

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

### 2.3 Other Environments (not yet CRIE-BT adapted)

| Task | File | Description |
|---|---|---|
| Cabinet | `task_cabinet.py` | Open cabinet door, place objects inside |
| Sort | `task_sort.py` | Sort objects by category into separate zones |
| Sweep | `task_sweep.py` | Cooperative sweeping of scattered objects |
| Rope | `task_rope.py` | Bimanual rope manipulation |

---

## 3. Metrics

### 3.1 Primary Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Action Success Rate (ASR)** | `episodes where success=True / n` | The LLM produced a parseable EXECUTE block **and** the RRT executor physically completed the action. Primary metric for `open_loop` (where task completion is structurally ~0%). |
| **Task Completion Rate (TCR)** | `episodes where sim_success=True / n` | `env.get_reward_done()` returned done — the full task was assembled. Meaningful only for multi-step controllers (`direct_feedback`, `bt_mediated`). |
| **ASR 95% CI** | Wilson score interval | Tighter than normal approximation for small n. Reported as `[lo, hi]` in percent. |

> **Why ASR and not TCR for open_loop?** The open_loop controller executes exactly one LLM-planned action per episode. A sandwich needs ~10 actions. TCR is structurally ~0% regardless of LLM quality, so it cannot differentiate communication modes. ASR measures what the LLM actually controls: plan validity and physical executability of the first action.

### 3.2 Efficiency Metrics

| Metric | Key | Description |
|---|---|---|
| Steps | `avg_steps_all ± std` | Total executor `.step()` calls per episode (one call = one simulator tick) |
| Wall time | `avg_wall_time_s ± std` | Total episode wall-clock time in seconds |
| LLM latency | `avg_llm_latency_s` | Mean per-call LLM response time (excludes RRT/sim time) |
| Planner errors | `planner_errors` (count/n) | Episodes that exhausted all `num_replans` retries without a valid EXECUTE block |
| Replans | `avg_replans` | Mean number of LLM replan attempts per episode |
| Local retries | `avg_local_retries` | Mean executor-level retries (sub-LLM recovery) |

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

## 4. Current Results Summary (open_loop, n=15/mode, Gemini 2.5 Flash)

| Mode | ASR (%) | 95% CI | Steps | Time (s) | LLM Lat (s) | P.Err |
|---|---|---|---|---|---|---|
| Plan (Centralised) | 100.0 | [80, 100] | 1.0 ± 0.0 | 9.3 ± 0.3 | 4.9 | 0/15 |
| Chat (w/ History) | 100.0 | [80, 100] | 1.0 ± 0.0 | 15.6 ± 1.6 | 12.5 | 0/15 |
| Dialog (Multi-agent) | 86.7 | [62, 96] | 0.9 ± 0.4 | 18.5 ± 11.7 | 16.0 | 2/15 |

Raw data: `results/sandwich_open_loop/episodes.jsonl`  
LaTeX table: `results/sandwich_open_loop/table.tex`
