# Environment Description

We evaluate multi-robot collaboration in a RoCoBench PackGrocery environment with two embodied agents, Alice and Bob. Alice controls a UR5e arm with a Robotiq gripper, while Bob controls a Franka Panda arm. The workspace contains grocery objects, target bin slots, and distractor objects. Each episode requires the agents to place assigned objects into assigned slots while respecting task constraints such as object ownership, target-slot occupancy, workspace safety, and maximum step limits.

The benchmark is organized around tasks, variations, subtasks, and skills. A task defines the high-level objective, active agents, success predicate, failure predicates, action representation, camera requirements, and maximum episode length. For example, a single-agent task may require Alice to place one assigned object into one assigned bin slot, while a two-agent task may require Alice and Bob to complete disjoint packing objectives either sequentially or under a conservative concurrent-execution rule.

A variation is a deterministic instance of a task. It fixes the random seed, participating agents, object-to-agent assignments, target slots, distractor set, object pose set, concurrency case, and expected skill plan. Variations allow different methods to be compared on the same concrete episode rather than on loosely matched task descriptions. Each variation is assigned a stable identifier and hash so benchmark results can be paired and audited by task-variation key.

A skill is a typed high-level robot capability exposed to the planner. Skills abstract away low-level joint commands, waypoints, and simulator control details. In the current PackGrocery benchmark, the planner-visible skills are `PUT_OBJECT_IN_CONTAINER(object, container)` and `WAIT()`. The first skill instructs an agent to pick the specified grocery object if necessary, transport it to the specified bin slot, place it, and release it. The second keeps an agent stationary while another agent acts. Skill calls are written in a structured per-agent grammar, for example:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

A subtask is one intended unit of progress inside a task plan. Operationally, a subtask corresponds to one non-wait skill call in a variation's expected skill plan. A single-agent packing variation therefore contains one packing subtask, while a two-agent packing variation contains two packing subtasks, one per assigned agent-object-slot tuple. In the current implementation, subtasks are represented directly as entries in `expected_skill_plan` rather than as a separate benchmark class.

At execution time, a high-level skill plan is parsed and validated before being routed to an executor. Validation checks that each configured agent has exactly one skill call, that skill arguments are well formed, that referenced objects and target slots exist, and that agents do not claim conflicting objects or slots. Executor backends then compile accepted skill calls into lower-level behavior. The RRT backend produces simulator-native motion plans and `SimAction` commands, while bridge-backed policies interact through a Gymnasium-style action vector that is converted to `SimAction` inside the RoCo simulator process.

The environment also supports grounded feedback for planner repair. After a parse failure, validation failure, or execution failure, the system renders structured feedback containing the failed skill, failure code, executor reason, measured state facts, inferred explanations, and remaining retry or fallback budget. This feedback is inserted into the next planner prompt as the previous structured outcome. The planner can then produce a revised skill plan from the current or restored simulator state. This loop provides a bounded mechanism for replanning, fallback, and recovery while preserving auditable task and variation definitions.

In the current benchmark release, the manifest contains four PackGrocery tasks and eight fixed variations. These include two single-agent skill-policy tasks, one for Alice and one for Bob, plus two two-agent planner-skill tasks covering sequential and conservative safe-concurrent packing. The final benchmark design can extend this same structure to six tasks and richer multi-subtask plans by adding task declarations and variation entries while preserving the same definitions of skills, subtasks, variations, and planner feedback.
