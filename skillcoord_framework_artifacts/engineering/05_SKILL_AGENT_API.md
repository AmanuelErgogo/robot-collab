# 05 — Skill and Agent API Specification

## AgentSpec example

```yaml
name: Alice
type: robot
embodiment: panda_left
capabilities:
  - PICK_OBJECT
  - PLACE_OBJECT
  - PUT_OBJECT_IN_CONTAINER
  - WAIT
workspace:
  zones: [left, center, bin_left]
constraints:
  max_concurrent_skills: 1
  safety_radius_m: 0.10
```

## HumanAgentSpec example

```yaml
name: Human
type: simulated_human
capabilities:
  - READ_LABEL
  - CONFIRM_TARGET
  - PICK_OBJECT
  - HOLD_CONTAINER
  - WAIT
response_model:
  delay_distribution: lognormal
  mean_delay_s: 2.0
  mistake_probability: 0.05
  unavailable_probability: 0.03
preferences:
  avoid_redundant_work: true
  prefer_low_workload: true
```

## SkillSpec example

```yaml
name: PUT_OBJECT_IN_CONTAINER
description: Pick an object if needed and place it inside a target container.
arguments:
  object:
    type: object
    required: true
  container:
    type: receptacle
    required: true
supported_agent_types: [robot, simulated_human, real_human]
preconditions:
  - object_exists(object)
  - container_exists(container)
  - not target_occupied(container)
postconditions:
  - object_inside(object, container)
  - gripper_or_hand_empty(agent)
resources:
  exclusive:
    - object:{object}
    - container:{container}
  workspace:
    - corridor(agent, object, container)
failure_modes:
  - MISSED_GRASP
  - SLIPPAGE
  - TARGET_OCCUPIED
  - TIMEOUT
  - SAFETY_CONFLICT
backends: [rrt, act, simulated_human, teleop]
```

## SkillCall

```json
{"agent":"Alice","skill":"PUT_OBJECT_IN_CONTAINER","arguments":{"object":"apple","container":"bin_front_left"}}
```

## SkillResult

```json
{
  "agent": "Alice",
  "skill": "PUT_OBJECT_IN_CONTAINER",
  "success": false,
  "status": "failed",
  "failure_code": "SLIPPAGE",
  "progress_stage": "transporting",
  "evidence": {"object": "apple", "last_known_location": "near bin_front_left"},
  "recovery_recommendation": "REPLAN_FROM_CURRENT_STATE",
  "backend": "act",
  "policy_id": "alice_pack_act_v1",
  "steps": 84
}
```

## Interfaces

```python
class Planner:
    def plan(self, state, goal, context):
        """Return a typed SkillPlan."""

class SkillBackend:
    def can_execute(self, call, state):
        """Return capability/precondition result."""
    def execute(self, call, context):
        """Return SkillResult."""

class Scheduler:
    def schedule(self, plan, state):
        """Return sequential/concurrent/reject decision."""
```
