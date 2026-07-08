# Metrics and Logging Specification

## Required episode-level fields

Every row in `results.jsonl` should contain:

```json
{
  "episode_id": "step1_sandwich_seed0_ep000_vlm_rr_cent",
  "stage": "step1",
  "condition_name": "VLM-RR-Cent",
  "controller_family": "VLM",
  "team_type": "RR",
  "coordination_mode": "Cent",
  "skill_backend": "RRT",
  "monitor_backend": "VLM-self",
  "monitor_privileged": false,
  "environment": "sim",
  "task_id": "sandwich",
  "seed": 0,
  "episode_index": 0,
  "success": true,
  "task_done": true,
  "completion_time_s": 42.1,
  "num_steps": 12,
  "planner_calls": 3,
  "replans": 1,
  "local_retries": 0,
  "failed_subtasks": 0,
  "dialogue_turns": 2,
  "human_interventions": 0,
  "monitor_updates": 0,
  "llm_prompt_tokens": 0,
  "llm_completion_tokens": 0,
  "llm_total_tokens": 0,
  "wall_time_s": 48.5
}
```

## Core task metrics

| Metric | Definition |
|---|---|
| task success | final task done according to environment/evaluator. |
| completion time | wall-clock or simulated execution time until task done/fail. |
| number of steps | number of controller/executor steps. |
| planner calls | number of VLM/LLM calls. |
| replans | number of plan revisions after the initial plan. |
| local retries | retries handled without a full planner call. |
| failed subtasks | number of subtasks whose executor returns failure/timeout. |
| recovery success | failure occurred and task eventually completed. |
| recovery latency | time or steps from failure/progress stop to resumed progress or stage completion. |
| unnecessary replans | replan was triggered while ground-truth evaluator says progress was still being made. |

## Monitor metrics

Monitor metrics apply directly to CRIE-BT because it has an explicit monitor. For the baseline, self-monitoring can be evaluated indirectly by extracting planner decisions, but the baseline does not have a modular monitor.

| Metric | Definition |
|---|---|
| progress score MAE | absolute error between monitor progress score and normalized ground-truth stage progress, when labels exist. |
| stage done precision | predicted stage done and ground-truth stage done. |
| stage done recall | ground-truth stage done detected by monitor. |
| stuck/replan precision | monitor triggered replan when evaluator says progress was actually stopped or failed. |
| stuck/replan false positive rate | monitor triggered replan while progress was still being made. |
| detection latency | steps between true progress stop/failure and monitor detection. |
| monitor update frequency | number of monitor calls per episode. |

## Baseline self-monitoring metrics

Since the baseline has no separate monitor, use behavior-level proxy metrics:

| Metric | Definition |
|---|---|
| self-monitor replan accuracy | VLM replans after actual failure/stuck versus unnecessary replans. |
| missed failure rate | failure/stuck occurs but VLM continues with invalid plan. |
| redundant replanning | VLM replans repeatedly without task-state change. |
| invalid next action rate | planner selects impossible or already-completed action. |

## Human-robot metrics

| Metric | Definition |
|---|---|
| dialogue turns | number of robot-human or robot-robot messages. |
| human instruction count | number of explicit human instructions. |
| human accept/reject/counter-proposal counts | responses from terminal interface. |
| human response time | time from instruction/message to response. |
| human idle time | time human has no useful action while task remains incomplete. |
| robot idle time | time robot waits for human or planner while executable work exists. |
| coordination conflict count | both agents attempt incompatible actions or target same object/space. |
| intervention count | number of manual corrections or overrides. |
| dialogue burden | dialogue turns or words per completed task/stage. |

## Learned-skill metrics for Step 3

| Metric | Definition |
|---|---|
| learned skill success | per-skill success rate. |
| learned skill execution time | time from skill start to success/failure/timeout. |
| SARM progress MAE | error between SARM progress score and annotated progress label. |
| SARM stage-done accuracy | whether SARM correctly detects stage completion. |
| safe stop count | number of safety stops. |
| fallback count | number of fallbacks to human/manual/RRT, if applicable. |

## Event log schema

Each episode should also save `events.jsonl` with event-level rows:

```json
{
  "timestamp": 12.34,
  "episode_id": "step1_sandwich_seed0_ep000_criebt_rr_cent",
  "event_type": "monitor_update",
  "condition_name": "CRIE-BT-RR-Cent",
  "agent_id": "robot_1",
  "stage_id": "pick_tomato",
  "skill_name": "pick_place",
  "planner_call_id": "call_003",
  "bt_node_id": "node_007",
  "execution_status": "running",
  "monitor_status": "in_progress",
  "progress_score": 0.42,
  "should_replan": false,
  "message": "Object is near gripper.",
  "privileged_evidence_used": true
}
```

Recommended event types:

- `episode_start`
- `planner_call_start`
- `planner_call_end`
- `bt_tick`
- `skill_start`
- `skill_feedback`
- `skill_success`
- `skill_failure`
- `monitor_update`
- `replan_triggered`
- `local_retry`
- `human_instruction`
- `human_response`
- `dialogue_message`
- `episode_end`

## Key fairness logs

Always log:

- whether monitor used privileged simulator information;
- whether planner received oracle state;
- whether planner input was image-only, image+status, or oracle text;
- whether human was controlled by terminal, scripted policy, or real participant.
