# CRIE-BT: Full Experimental Design

## 1. Conditions

Each condition is a cross of a **feedback mode** (how failure is handled) × a **communication mode** (how agents coordinate via LLM).

### 1.1 Feedback Modes

| Label | Controller | Description |
|---|---|---|
| **No-Feedback** | `OpenLoopController` | LLM plans once at episode start. All steps execute sequentially. No recovery on failure. Baseline. |
| **With-Feedback** | `DirectFeedbackController` | After each failed step, failure details are returned to the LLM which replans. Loop until success or step budget. |
| **Feedback+BT** *(our method)* | `BTMediatedController` | Failure handling is governed by a Behavior Tree. The BT reads uncertainty estimates and decides whether to replan, retry locally, or escalate to a human. Adds structured recovery beyond direct replanning. |

### 1.2 Communication Modes

| Label | Prompter | Description |
|---|---|---|
| **Centralised (w/ history)** | `SingleThreadPrompter` with history | A single prompt covering all agents, with the full prior conversation appended. One LLM call per planning round. |
| **Dialog** | `DialogPrompter` | Agents take turns: each agent sends and receives one message per round before a joint action is committed. Richer but more expensive and prone to single-agent parse failures. |

> Note: a third mode, **Centralised (no history)** (`plan`), exists in the codebase and was used for initial debugging but is not a primary condition in the paper — it is subsumed by Centralised (w/ history).

### 1.3 Full Condition Matrix (6 primary conditions)

| | Centralised (w/ history) | Dialog |
|---|---|---|
| **No-Feedback** | C1 | C2 |
| **With-Feedback** | C3 | C4 |
| **Feedback+BT** *(ours)* | C5 | C6 |

Status:

| Condition | Status |
|---|---|
| C1 — No-Feedback × Centralised | ✅ Evaluated (n=15, sandwich, Gemini 2.5 Flash) |
| C2 — No-Feedback × Dialog | ✅ Evaluated (n=15, sandwich, Gemini 2.5 Flash) |
| C3 — With-Feedback × Centralised | ⚙️ Controller implemented, not yet evaluated |
| C4 — With-Feedback × Dialog | ⚙️ Controller implemented, not yet evaluated |
| C5 — Feedback+BT × Centralised | ⚙️ Controller implemented, not yet evaluated |
| C6 — Feedback+BT × Dialog | ⚙️ Controller implemented, not yet evaluated |

---

## 2. Agent Configurations

The same 6 conditions run under four agent configurations. In all configurations, **"human" means a human operator selects the next subtask** (e.g. "PICK bread_slice1") and the robot executes it autonomously using its low-level skill executor (RRT motion planning). The human replaces the LLM planner — the execution pipeline (skill executor, BT, feedback) remains identical.

| Configuration | Planners | Executors | Role |
|---|---|---|---|
| **Robot–Robot** | LLM + LLM | Arm + Arm | Both subtask selections made by LLM. Primary configuration for all 6 conditions. |
| **Human–Robot** | Human + LLM | Arm + Arm | One operator selects subtasks for one arm; the other arm is LLM-planned. Tests asymmetric teaming where the LLM must adapt to a human partner's choices. |
| **Human–Human** | Human + Human | Arm + Arm | Both subtask selections made by human operators. **Upper-bound baseline**: shows peak task completion when perfect coordination is possible. No LLM involved. Applies to all tasks. |
| **Single Robot** | LLM (or Human) | Arm only | One arm, one planner. Tests individual capability in isolation. Only feasible for tasks a single arm can physically complete. |

> **Human teleoperation interface**: The human is presented with the current observation and selects a subtask from the valid action set (e.g. `PICK <obj>`, `PLACE <target>`, `WAIT`). The selected subtask is passed directly to `LegacyTaskRRTExecutorAdapter` — the same execution path used by the LLM. This means Human–Human provides a fair upper bound because it uses identical low-level execution. Human–Human does **not** apply to conditions C1–C6 (which all require an LLM planner); it is a separate reference condition evaluated once per task.

---

## 3. Tasks

Six tasks, evaluated on all 6 conditions × 3 agent configurations where feasible. Tasks are classified by their **coordination structure** and **ordering constraint**, since these determine which failure modes are most likely and which recovery strategies matter.

### Task Classification

| Class | Definition | Expected difficulty |
|---|---|---|
| **Sequential-Dependent** | Steps must happen in a fixed order; each step depends on the previous one. | High: wrong order = unrecoverable without full reset. BT recovery most valuable. |
| **Parallel-Independent** | Steps can happen in any order and simultaneously. | Medium: coordination needed to avoid collision but no strict sequencing. |
| **Gated** | One agent must complete a prerequisite before the other can proceed. | Medium-High: deadlock possible if gate is not recognised. |
| **Tightly-Coupled** | Both agents must act simultaneously on the same object. | Very high: requires precise synchronisation; dialog mode most critical. |
| **Continuous-Cooperative** | Task has no discrete steps; agents must maintain a shared state over time. | High for replanning approaches; no natural plan granularity. |

### Task Table

| Task | File | Agents (R-R) | Class | Ordering | CRIE-BT adapted |
|---|---|---|---|---|---|
| **Sandwich** | `task_sandwich.py` | Chad + Dave (or humanoid) | Sequential-Dependent | Strict: bread→bacon→cheese→tomato→bread | ✅ Full |
| **Pack Grocery** | `task_pack.py` | Alice + Bob | Parallel-Independent | Flexible: pack any item at any time | ✅ Partial |
| **Cabinet** | `task_cabinet.py` | 3 agents (Alice, Bob, Carol) | Gated | One opens cabinet, others place items inside | ❌ Not adapted |
| **Sort** | `task_sort.py` | 3 agents (Alice, Bob, Carol) | Parallel-Independent | Flexible: sort any object to correct bin | ❌ Not adapted |
| **Sweep** | `task_sweep.py` | Chad + Dave | Continuous-Cooperative | None: continuous sweeping motion | ❌ Not adapted |
| **Rope** | `task_rope.py` | Chad + Dave | Tightly-Coupled | Simultaneous: both grip rope ends | ❌ Not adapted |

> Cabinet and Sort use **3 agents**, which Dialog mode handles naturally (additional turn per agent) but Centralised mode must encode all three in one prompt.

---

## 4. Metrics

### 4.1 Per-Episode Outcomes

| Metric | Key | Definition |
|---|---|---|
| **Action Success Rate (ASR)** | `action_success_rate` | % episodes where the LLM produced a valid EXECUTE block **and** the RRT physically executed it. Primary metric for No-Feedback conditions where task completion is structurally low. |
| **Task Completion Rate (TCR)** | `task_completion_rate` | % episodes where `env.get_reward_done()` returned `done=True` (full task assembled). Primary metric for With-Feedback and Feedback+BT conditions. |
| **ASR 95% CI** | `asr_ci_lo / asr_ci_hi` | Wilson score interval. Tighter than normal approximation for small n. |

> **Why both?** For No-Feedback (open_loop), TCR ≈ 0% by construction (one LLM call cannot complete a ~10-step task), so ASR is the discriminating metric. For With-Feedback and Feedback+BT, the controller loops until done or timeout, making TCR the meaningful outcome. Both are always reported.

### 4.2 Efficiency Metrics

| Metric | Key | Description |
|---|---|---|
| Steps | `avg_steps_all ± std` | Executor `.step()` calls per episode (≈ simulator ticks) |
| Wall time | `avg_wall_time_s ± std` | Total episode clock time (seconds) |
| LLM latency | `avg_llm_latency_s` | Mean per-call LLM response time; excludes sim/RRT |
| Planner errors | `planner_errors` (count/n) | Episodes that exhausted `num_replans` without a valid EXECUTE block |
| Replans | `avg_replans` | Mean LLM replan calls per episode |
| Local retries | `avg_local_retries` | Mean executor-level retries below the LLM layer |

### 4.3 Recovery Metrics (With-Feedback and Feedback+BT only)

| Metric | Definition |
|---|---|
| **Recovery Rate** | Among episodes that had at least one failure, fraction that ultimately succeeded |
| **Steps-to-Recovery** | Mean steps taken from first failure to next success |
| **Unnecessary Replans** | Replans triggered when no actual failure occurred (proxy for over-sensitivity) |

### 4.4 Failure Codes

Every episode logs `failure_counts` broken down by `FailureCode`:

| Code | Trigger |
|---|---|
| `PLANNER_ERROR` | LLM never returned a parseable EXECUTE block within `num_replans` retries |
| `POSTCONDITION_FAILED` | RRT ran but skill postcondition not satisfied (e.g. object not grasped) |
| `MISSED_GRASP` | Gripper closed but object not detected as held |
| `NO_PROGRESS` | Executor exceeded sim step budget without completing |
| `WRONG_OBJECT` | Robot picked the wrong object |
| `WRONG_TARGET` | Robot placed at the wrong target location |
| `SAFETY_CONFLICT` | Planned action would cause collision |
| `TIMEOUT` | Episode exceeded wall-clock budget |
| `UNKNOWN` | Unclassified executor exception |

---

## 5. Full Experiment Matrix

Rows = conditions, Columns = configurations (4), Cells = tasks evaluated. Human–Human is a reference baseline, not a condition of C1–C6.

|  | Robot–Robot | Human–Robot | Human–Human | Single Robot |
| --- | --- | --- | --- | --- |
| **C1** No-Feedback × Cent | Sandwich ✅, Pack, Cabinet, Sort, Sweep, Rope | Sandwich, Pack | — | Sandwich, Pack |
| **C2** No-Feedback × Dialog | Sandwich ✅, Pack, Cabinet, Sort, Sweep, Rope | Sandwich, Pack | — | Sandwich, Pack |
| **C3** With-Feedback × Cent | Sandwich, Pack, Cabinet, Sort, Sweep, Rope | Sandwich, Pack | — | Sandwich, Pack |
| **C4** With-Feedback × Dialog | Sandwich, Pack, Cabinet, Sort, Sweep, Rope | Sandwich, Pack | — | Sandwich, Pack |
| **C5** Feedback+BT × Cent *(ours)* | Sandwich, Pack, Cabinet, Sort, Sweep, Rope | Sandwich, Pack | — | Sandwich, Pack |
| **C6** Feedback+BT × Dialog *(ours)* | Sandwich, Pack, Cabinet, Sort, Sweep, Rope | Sandwich, Pack | — | Sandwich, Pack |
| **Human–Human** *(reference)* | — | — | All 6 tasks | Sandwich, Pack |

> Single Robot is only feasible for tasks one arm can complete (Sandwich, Pack). Sweep and Rope require two arms by design. Human–Human across all 6 tasks establishes the performance ceiling for each task under ideal coordination.

---

## 6. Implementation Gaps

| Gap | Priority | Notes |
|---|---|---|
| CRIE-BT adapters for Cabinet, Sort, Sweep, Rope | High | Need `LegacyPromptPlanner` + `LegacyTaskRRTExecutorAdapter` per task |
| Human–Robot mode | Medium | Needs scripted humanoid policy; sandwich env already has humanoid body |
| Single-Robot mode | Medium | Need single-agent prompters and executor adapters |
| Evaluate C3–C6 on sandwich | High | Controllers exist; need eval script extension |
| More episodes for tighter CIs | Medium | Current n=15; paper typically needs n=30–50 |
| Fix RRT grasp inflation | Medium | RRT reports `success=True` (TRANSPORTING stage) even when object not grasped |
