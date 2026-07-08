# Environment Construction Tutorial

This tutorial explains how the current RoCoBench PackGrocery environment is
constructed and how its benchmark pieces fit together. It is intended for
paper authors and implementers who need a concrete view of tasks, variations,
skills, subtasks, agents, runtime state, and planner feedback.

## 1. Start With The Environment Domain

The current benchmark domain is PackGrocery. Two robot agents operate in one
MuJoCo scene:

- Alice controls a UR5e arm with a Robotiq gripper.
- Bob controls a Franka Panda arm.

The scene contains grocery objects, target bin slots, and distractors. The
high-level objective is to move assigned grocery objects into assigned target
slots while avoiding invalid plans, unsafe states, object loss, and resource
conflicts.

The simulator remains owned by the RoCo Python 3.8 runtime. Gymnasium or
LeRobot clients interact with the simulator through the Phase 0 bridge rather
than importing `rocobench` directly.

## 2. Understand The Four Core Constructs

The environment is built from four related constructs:

```text
Task
  Variation
    Subtask / expected skill-plan entry
      Skill call
```

### Task

A task is the benchmark-level objective template. It defines:

- task ID;
- natural-language description;
- active agents;
- maximum episode steps;
- success and failure predicates;
- action representation;
- required cameras;
- variation group.

Tasks are declared in:

```text
benchmark/manifests/pack_v1_tasks.json
```

The current release has four tasks:

```text
pack.put.alice
pack.put.bob
pack.sequential.two_agent
pack.concurrent.safe_two_agent
```

The first two are single-agent skill-policy tasks. The latter two are
two-agent planner-skill tasks.

### Variation

A variation is a deterministic episode instance for a task. It fixes:

- variation ID;
- variation group;
- seed;
- active agents;
- object assignments;
- target slots;
- distractors;
- pose set;
- concurrency case;
- expected skill plan;
- optional metadata.

Variations are declared in:

```text
benchmark/manifests/pack_v1_variations.json
```

Each variation has a stable hash. Results use `task_id::variation_id` keys for
paired comparison, so two methods can be compared on the exact same episode.

### Skill

A skill is a typed high-level robot capability exposed to the planner. It is
not a low-level trajectory or actuator command. The current PackGrocery skill
registry exposes:

```text
PUT_OBJECT_IN_CONTAINER(object, container)
WAIT()
```

Skills are defined in:

```text
rocobench/skills/pack_grocery.py
rocobench/skills/models.py
```

A planner writes skill calls using one line per configured agent:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

The parser turns this text into a `SkillPlan`, which contains one `SkillCall`
per agent.

### Subtask

A subtask is one intended unit of progress inside a larger task plan. In the
current benchmark, subtasks are represented directly as entries in a
variation's `expected_skill_plan`; there is no separate `BenchmarkSubtask`
class.

Example one-subtask variation:

```json
[
  {
    "agent_name": "Alice",
    "skill_name": "PUT_OBJECT_IN_CONTAINER",
    "object": "apple",
    "container": "bin_front_left"
  }
]
```

Example two-subtask variation:

```json
[
  {
    "agent_name": "Alice",
    "skill_name": "PUT_OBJECT_IN_CONTAINER",
    "object": "apple",
    "container": "bin_front_left"
  },
  {
    "agent_name": "Bob",
    "skill_name": "PUT_OBJECT_IN_CONTAINER",
    "object": "banana",
    "container": "bin_front_right"
  }
]
```

For paper language, it is accurate to say that a subtask corresponds to a
non-wait skill call in the plan.

## 3. Inspect The Current Environment

From the repository root, inspect task declarations:

```bash
python - <<'PY'
import json
with open("benchmark/manifests/pack_v1_tasks.json") as f:
    tasks = json.load(f)["tasks"]
for task in tasks:
    print(task["task_id"], task["active_agents"], task["action_representation"])
PY
```

Inspect variation and subtask counts:

```bash
python - <<'PY'
import json
from collections import Counter
with open("benchmark/manifests/pack_v1_variations.json") as f:
    variations = json.load(f)["variations"]
print("variations:", len(variations))
print("subtasks per variation:", Counter(len(v["expected_skill_plan"]) for v in variations))
for variation in variations:
    print(variation["variation_id"], variation["active_agents"], variation["expected_skill_plan"])
PY
```

At the time of this tutorial, the current release contains:

```text
4 tasks
8 variations
2 agents: Alice and Bob
2 planner-visible skills: PUT_OBJECT_IN_CONTAINER and WAIT
1 or 2 packing subtasks per variation
```

## 4. How A Bridge-Backed Episode Runs

Bridge-backed policies use the Phase 0 client/server boundary:

```text
Gymnasium / LeRobot runtime
  RoCoGymEnv
  action: float32 vector
        |
        | ZeroMQ REQ/REP, msgpack, ndarray-safe encoding
        v
RoCo Python 3.8 runtime
  RoCoBridgeServer
  RoCoActionAdapter
  SimAction
  PackGroceryTask.step(...)
        |
        v
observation, reward, terminated, truncated, info["is_success"]
```

The raw observation contract is:

```text
pixels.front
pixels.active_agent
agent_pos
```

The bridge server owns simulator state. The client mirrors Gym episode state,
checks action shapes and bounds, and always expects `info["is_success"]`.

The action sent by the client is not a `SimAction`. The server converts the
client action vector into a simulator-native `SimAction`, appending hold
controls for passive agents when needed.

## 5. How A Skill Plan Runs

A planner-skill episode follows a higher-level path:

```text
state observation
  planner prompt
  planner response text
  SkillResponseParser
  SkillPlan
  PackGrocerySkillPlanValidator
  executor router
  learned policy or RRT backend
  simulator execution
  success / failure / feedback
```

Validation checks include:

- one call per configured agent;
- no duplicate agent calls;
- known agents;
- known skill names;
- required skill arguments;
- known objects and target slots;
- no object already packed;
- no occupied target slot;
- no two agents claiming the same object or target;
- at least one non-wait progress skill.

The RRT backend compiles accepted skill calls into simulator-native motion
plans and eventually into `SimAction` commands. Learned policies use the
bridge-compatible observation and action representation.

## 6. How Feedback Reaches The Planner

The planning controller supports bounded feedback and replanning. On parse,
validation, or execution failure, it renders structured feedback with:

- failed agent and skill;
- status and failure code;
- executor reason;
- measured state facts;
- inferred explanation when available;
- retry, fallback, and replan budget;
- deterministic recovery action.

The next prompt includes this text as the previous structured outcome:

```text
[Previous Structured Outcome]
Status: PLAN_REJECTED
Validation issues:
- [TARGET_OCCUPIED] bin_front_left is already occupied by banana.
Measured current state:
- apple position is ...
- bin_front_left is occupied by banana.
Retry budget: ...
Choose one valid skill plan from the available capabilities.
```

This feedback loop is implemented in:

```text
rocobench/planning/run_controller.py
rocobench/planning/feedback_renderer.py
rocobench/planning/replanning_policy.py
```

The Phase 8 demo currently exercises fixed-manifest hold and RRT baselines, not
the full planner feedback loop. The feedback loop is available as the Phase 6
planner integration path.

## 7. How Benchmark Outputs Are Constructed

The evaluator loads tasks and variations, runs one baseline per selected
task-variation pair, and writes normalized outputs:

```text
run_manifest.json
episodes.csv
metrics.json
report.md
episodes/<episode_id>/result.json
episodes/<episode_id>/benchmark_result.json
```

Every episode result records:

- benchmark, schema, protocol, and predicate versions;
- variation manifest hash;
- task ID and variation ID;
- variation hash;
- track and method;
- success, overall success, learned success, and fallback success;
- termination reason;
- step counts and timing;
- planner and multi-agent fields when relevant;
- artifact paths and extra JSON.

This keeps bridge-backed learned-policy runs and direct RoCo RRT runs
comparable at the benchmark-result layer.

## 8. Extending Toward A Six-Task Environment

To extend the current environment to a final six-task benchmark, keep the same
constructs and add declarations rather than changing runtime contracts.

### Add a task

Add a task object to:

```text
benchmark/manifests/pack_v1_tasks.json
```

Each task should define a new `task_id` and point to a `variation_group`.

### Add variations

Add one or more matching variation objects to:

```text
benchmark/manifests/pack_v1_variations.json
```

Each variation should fix concrete object assignments, target slots, seed,
pose set, and an `expected_skill_plan`.

### Add richer subtasks

For multi-subtask tasks, add multiple `expected_skill_plan` entries. If the
task requires temporal ordering beyond one skill per agent, introduce an
explicit ordered subtask schema instead of overloading a single concurrent
`SkillPlan`.

### Add skills only when needed

If new tasks require behavior beyond packing and waiting, add new `SkillSpec`
entries to the skill registry and update:

```text
rocobench/skills/pack_grocery.py
rocobench/skills/validation.py
prompting/skill_parser.py
```

New skills should have typed arguments, validation rules, resource claims, and
an execution backend before they appear in benchmark manifests.

### Preserve the runtime boundary

Do not import Gymnasium or LeRobot into normal RoCo modules. Do not import
`rocobench` from the isolated client package. The simulator state should remain
server-owned, with client-side episode state mirrored through the bridge
protocol.

## 9. Minimal Mental Model

Use this compact model when writing or debugging the environment:

```text
Task: what objective family is being evaluated?
Variation: which concrete seed, objects, slots, and distractors?
Subtask: which unit of progress must be completed?
Skill: what typed high-level capability asks a robot to complete that subtask?
Executor: how does the skill become actions in the simulator?
Feedback: what measured failure information returns to the planner?
```

The current release already has this structure, but the scored Phase 8 demo is
still conservative: it uses fixed expected plans and baseline execution. The
full final environment can grow by adding task and variation manifests,
expanding subtask plans, and connecting the planner feedback loop into the
benchmark evaluation path.
