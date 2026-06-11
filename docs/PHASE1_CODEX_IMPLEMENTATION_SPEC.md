# Phase 1 Implementation Package
## Skill-Level Planning Interface with the Existing RoCo RRT Backend

**Repository:** `AmanuelErgogo/robot-collab`  
**Initial supported task:** `PackGroceryTask` (`--task pack`)  
**Target Python compatibility:** Python 3.8  
**Phase objective:** Add a typed, extensible skill-planning and skill-execution layer while retaining the existing `LLMPathPlan` + `PlannedPathPolicy` + `MultiArmRRT` stack as the first execution backend.

---

# 1. Executive Summary

Phase 1 is an architectural refactor, not an imitation-learning implementation.

The current execution path is approximately:

```text
LLM response
    -> LLMResponseParser
    -> List[LLMPathPlan]
    -> FeedbackManager
    -> PlannedPathPolicy
    -> MultiArmRRT
    -> env.step(...)
```

Phase 1 must add this path:

```text
LLM response containing named skills
    -> SkillResponseParser
    -> SkillPlan
    -> SkillPlanValidator
    -> RRTSkillCompiler
    -> existing List[LLMPathPlan]
    -> existing geometric FeedbackManager
    -> RRTSkillExecutor
    -> existing PlannedPathPolicy / MultiArmRRT
    -> env.step(...)
```

The skill-level representation must survive all the way to the runner and executor. Do **not** parse a skill response and immediately return only `LLMPathPlan` objects; that would hide the abstraction needed for the future LeRobot executor.

For Phase 1, support these task-level skills for both Alice and Bob:

```text
PUT_OBJECT_IN_CONTAINER(object=<grocery_item>, container=<bin_slot>)
WAIT()
```

Example final plan:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

The existing RRT implementation remains the online executor. A later phase will add:

```text
LearnedSkillExecutor(SkillExecutor)
```

without changing the LLM plan representation.

---

# 2. Goals

## 2.1 Functional goals

1. Add `skill` as a valid `--output_mode`.
2. Support skill-level planning for `PackGroceryTask`.
3. Parse strict, structured skill calls for all configured agents.
4. Validate skill names, arguments, agent capabilities, task state, and inter-agent resource conflicts.
5. Compile valid skill plans into the existing action-only RoCo representation.
6. Reuse the existing `LLMResponseParser`, `FeedbackManager`, `PlannedPathPolicy`, and `MultiArmRRT` rather than duplicating geometry, IK, collision, or motion-planning logic.
7. Execute the compiled plan through a `SkillExecutor` abstraction.
8. Save skill-level and compiled execution artifacts.
9. Preserve the exact behavior of existing `action_only` and `action_and_path` modes.
10. Add unit tests and a no-LLM smoke test.

## 2.2 Architectural goals

1. Establish a stable typed contract between:
   - high-level planner,
   - skill registry,
   - semantic validator,
   - backend compiler,
   - execution backend.
2. Keep task semantics separate from geometric motion planning.
3. Make a future LeRobot executor replaceable through dependency injection.
4. Avoid import cycles between `prompting`, `rocobench.skills`, `rocobench.policy`, and environment modules.
5. Keep task-specific skill definitions isolated from generic skill infrastructure.

---

# 3. Non-Goals

Do not implement any of the following in Phase 1:

- LeRobot imports or dependencies.
- Dataset recording.
- ACT, Diffusion Policy, SmolVLA, or any learned-policy inference.
- MetaWorld support.
- New robot controllers.
- New inverse-kinematics or collision-checking implementations.
- Replacement or deletion of `MultiArmRRT`.
- Natural-language semantic parsing of unrestricted phrases.
- Concurrent learned-policy execution.
- A joint multi-arm neural policy.
- Support for all six RoCo tasks.
- Broad modernization of the OpenAI/Anthropic clients.
- Migration to Python 3.12.
- Large unrelated refactors or formatting changes.

The implementation should make later work easier, but Phase 1 must remain small enough to review and test.

---

# 4. Existing Code That Must Be Respected

Before coding, inspect at least:

```text
run_dialog.py
prompting/parser.py
prompting/feedback.py
prompting/dialog_prompter.py
prompting/single_thread_prompter.py
prompting/__init__.py
rocobench/subtask_plan.py
rocobench/policy.py
rocobench/__init__.py
rocobench/envs/base_env.py
rocobench/envs/task_pack.py
requirements.txt
```

Important current behavior:

- `run_dialog.py` creates `MultiArmRRT`.
- `LLMResponseParser.parse(...)` returns `List[LLMPathPlan]`.
- The prompter expects the parser to expose:
  ```python
  parse(obs, response) -> (success, message, plans)
  ```
- The prompter expects the feedback manager to expose:
  ```python
  give_feedback(plan) -> (ready_to_execute, feedback)
  ```
- The prompter logs:
  ```python
  plan.get_action_desp()
  ```
- `display_plan(...)` assumes an `LLMPathPlan` with `path_3d_list`.
- `PackGroceryTask` currently prompts for `PICK`, `PLACE`, and `PATH`.
- `PackGroceryTask.get_task_feedback(...)` expects legacy action strings containing `PICK` or `PLACE`.
- The legacy parser can expand a combined action such as:
  ```text
  PICK apple PLACE bin_front_left
  ```
  into multiple synchronized path plans.
- The runner rewinds the environment if any path plan fails.
- Legacy CLI modes must remain unchanged.

---

# 5. Required High-Level Design

## 5.1 Proposed files

Create this structure unless repository inspection reveals a compelling reason to adjust names:

```text
rocobench/
└── skills/
    ├── __init__.py
    ├── models.py
    ├── registry.py
    ├── validation.py
    ├── compiler.py
    ├── executor.py
    └── pack_grocery.py

prompting/
├── skill_parser.py
├── skill_feedback.py
└── skill_prompt.py

tests/
├── skills/
│   ├── test_skill_models.py
│   ├── test_skill_registry.py
│   ├── test_skill_parser.py
│   ├── test_pack_skill_validation.py
│   ├── test_rrt_skill_compiler.py
│   ├── test_skill_feedback.py
│   └── test_rrt_skill_executor.py
├── test_legacy_parser_regression.py
└── integration/
    └── test_pack_skill_smoke.py

scripts/
└── smoke_test_pack_skill.py

docs/
└── phase1_skill_architecture.md

requirements-dev.txt
pytest.ini
```

If the repository already has a preferred test or documentation layout, follow it and document the deviation.

## 5.2 Dependency direction

Use this dependency direction:

```text
prompting.skill_parser
    -> rocobench.skills.models
    -> rocobench.skills.registry

prompting.skill_prompt
    -> rocobench.skills.registry
    -> PackGroceryTask public/task attributes

prompting.skill_feedback
    -> rocobench.skills.validation
    -> rocobench.skills.compiler
    -> existing prompting.feedback.FeedbackManager

rocobench.skills.executor
    -> existing rocobench.policy.PlannedPathPolicy
    -> existing LLMPathPlan / MultiArmRRT
```

Avoid importing `prompting` from generic `rocobench.skills.models` or `registry`.

---

# 6. Domain Models

Use Python 3.8-compatible `dataclasses`, `Enum`, and typing forms such as `List[str]`, not `list[str]`.

## 6.1 `SkillSpec`

```python
@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    required_arguments: Tuple[str, ...]
    optional_arguments: Tuple[str, ...] = ()
    supported_agents: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    resource_arguments: Tuple[str, ...] = ()
```

Requirements:

- Store canonical names in uppercase snake case.
- Reject duplicate required/optional argument names.
- Reject overlap between required and optional arguments.
- Normalize aliases at registration.
- `supported_agents=()` may mean all configured agents, but document this behavior.

## 6.2 `SkillCall`

```python
@dataclass(frozen=True)
class SkillCall:
    agent_name: str
    skill_name: str
    arguments: Mapping[str, str]
    raw_action: str
```

Requirements:

- Normalize skill name to its canonical registry value.
- Copy arguments into an immutable or defensive representation.
- Preserve `raw_action` for diagnostics.
- Provide a deterministic `to_dict()`.
- Provide a canonical string representation:
  ```text
  PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
  ```

## 6.3 `SkillPlan`

```python
@dataclass
class SkillPlan:
    calls: List[SkillCall]
    parsed_proposal: str
    plan_id: str
    prepared_execution: Optional["PreparedSkillExecution"] = field(
        default=None,
        repr=False,
        compare=False,
    )
```

Requirements:

- One plan contains one call per configured agent.
- Calls must be stored in configured agent order, not arbitrary response order.
- `plan_id` must be deterministic for the same canonical plan, for example a short SHA-256 digest.
- Implement:
  ```python
  get_action_desp() -> str
  to_dict() -> Dict[str, Any]
  ```
- `get_action_desp()` must be suitable for existing prompt history and logs.
- Do not serialize runtime-only `prepared_execution`.

## 6.4 Validation models

```python
@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    agent_name: Optional[str] = None
    retryable: bool = True

@dataclass(frozen=True)
class SkillValidationResult:
    valid: bool
    issues: Tuple[ValidationIssue, ...] = ()
```

User-facing feedback should include stable error codes, for example:

```text
[UNKNOWN_SKILL] Alice cannot execute FLY_OBJECT.
[TARGET_OCCUPIED] bin_front_left is already occupied by banana.
[RESOURCE_CONFLICT] Alice and Bob both target apple.
```

## 6.5 Prepared execution model

```python
@dataclass
class PreparedSkillExecution:
    backend_name: str
    source_plan_id: str
    compiled_plans: List[LLMPathPlan]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

For Phase 1:

```text
backend_name = "rrt"
```

The generic `SkillPlan` must not need to know how `LLMPathPlan` works beyond holding this prepared backend object.

## 6.6 Execution result

```python
class SkillExecutionStatus(Enum):
    SUCCESS = "success"
    INVALID_PLAN = "invalid_plan"
    NOT_PREPARED = "not_prepared"
    MOTION_PLANNING_FAILED = "motion_planning_failed"
    EXECUTION_FAILED = "execution_failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"

@dataclass
class SkillExecutionResult:
    success: bool
    status: SkillExecutionStatus
    reason: str
    num_sim_steps: int
    reward: float
    done: bool
    info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Do not use raw unstructured tuples at the skill-executor boundary.

---

# 7. Skill Registry

Implement `SkillRegistry` with:

```python
class SkillRegistry:
    def register(self, spec: SkillSpec) -> None: ...
    def get(self, name_or_alias: str) -> SkillSpec: ...
    def has(self, name_or_alias: str) -> bool: ...
    def skills_for_agent(self, agent_name: str) -> List[SkillSpec]: ...
    def validate_call_shape(self, call: SkillCall) -> SkillValidationResult: ...
    def render_agent_skill_prompt(self, agent_name: str) -> str: ...
```

Requirements:

- Reject duplicate canonical names.
- Reject aliases colliding with canonical names or other aliases.
- Resolve names case-insensitively, but render canonical uppercase names.
- Validate missing, duplicate, and unknown arguments.
- Validate whether an agent supports a skill.
- Produce deterministic prompt ordering.
- Do not embed task-state validation in the generic registry.

## 7.1 Grocery registry

Provide:

```python
build_pack_grocery_skill_registry(agent_names: Sequence[str]) -> SkillRegistry
```

Register:

### `PUT_OBJECT_IN_CONTAINER`

```text
Arguments:
- object: required
- container: required

Resources:
- object
- container

Supported agents:
- Alice
- Bob
```

Semantics:

> Pick the specified grocery object if necessary, transport it, place it into the specified bin slot, and release it.

### `WAIT`

```text
Arguments: none
Resources: none
Supported agents: Alice and Bob
```

Semantics:

> Keep the robot stationary while another agent executes its skill.

Do not add extra skills during Phase 1 unless they are required by an existing code path and are covered by tests.

---

# 8. Structured Skill Grammar

## 8.1 Required final-output form

The planner may discuss before the final answer, but the executable section must be:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

## 8.2 Grammar

Conceptually:

```ebnf
execute_block = "EXECUTE", newline, agent_line, {newline, agent_line};
agent_line = "NAME", whitespace, agent, whitespace, "ACTION", whitespace, skill_call;
skill_call = skill_name, "(", [argument, {",", argument}], ")";
argument = identifier, "=", value;
```

Supported values:

- unquoted identifiers:
  ```text
  apple
  bin_front_left
  ```
- optionally, quoted strings:
  ```text
  "green bottle"
  ```
  Parse quoted strings safely and normalize them; never call `eval`.

## 8.3 Parser behavior

Implement `SkillResponseParser` with the same public shape as the existing parser:

```python
parse(obs, response) -> Tuple[bool, str, List[SkillPlan]]
```

The returned list should contain exactly one `SkillPlan` for the final coordinated round.

Requirements:

1. Ignore discussion text before `EXECUTE`.
2. Reject a response without `EXECUTE`.
3. Parse only executable lines after `EXECUTE`.
4. Require exactly one action for every configured agent.
5. Reject duplicate agent lines.
6. Reject unknown agents.
7. Reject unknown skills.
8. Reject malformed parentheses.
9. Reject positional arguments.
10. Reject missing `=`.
11. Reject duplicate argument keys.
12. Reject missing required arguments.
13. Reject unknown arguments.
14. Reject arguments passed to `WAIT`.
15. Return calls in configured agent order.
16. Never use `eval`, `exec`, or unsafe deserialization.
17. Return concise, actionable parsing errors that can be sent back to the LLM.
18. Do not raise expected formatting errors out of `parse`; return `False` and the message.
19. Log unexpected internal failures with a traceback and return a safe error message.

Examples that must parse:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

```text
Discussion...
EXECUTE
NAME Bob ACTION WAIT()
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(
    object=apple,
    container=bin_front_left
)
```

Supporting multiline calls is optional. If not supported, the prompt and error must explicitly require one action per line.

Examples that must fail:

```text
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(apple, bin_front_left)
NAME Bob ACTION WAIT()
```

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple)
NAME Bob ACTION WAIT()
```

```text
EXECUTE
NAME Alice ACTION UNKNOWN_SKILL()
NAME Bob ACTION WAIT()
```

```text
EXECUTE
NAME Alice ACTION WAIT()
NAME Alice ACTION WAIT()
```

---

# 9. Skill Prompting

The current grocery prompts contain explicit `PATH` instructions. Skill mode must not expose those instructions.

## 9.1 Prompt provider abstraction

Add an injectable prompt provider rather than hard-coding a skill-mode conditional throughout the prompters.

Suggested interface:

```python
class PlanningPromptProvider:
    def get_action_prompt(self, obs: EnvState, agent_name: str) -> str: ...
    def get_agent_prompt(self, obs: EnvState, agent_name: str) -> str: ...
```

Implement:

```python
class LegacyPlanningPromptProvider(...)
class PackGrocerySkillPromptProvider(...)
```

Alternatively, use two callables if that creates a smaller and cleaner diff. Preserve default behavior when no provider is supplied.

Modify both applicable prompter implementations so that:

```python
action_desp = provider.get_action_prompt(obs, agent_name)
agent_prompt = provider.get_agent_prompt(obs, agent_name)
```

falls back to:

```python
env.get_action_prompt()
env.get_agent_prompt(obs, agent_name)
```

for legacy mode.

## 9.2 Required grocery skill prompt

The skill prompt must include:

- task objective;
- agent identity;
- current object states;
- current robot holding states;
- valid grocery names;
- valid bin-slot names;
- skills available to that agent;
- exact argument names;
- skill preconditions;
- inter-agent coordination rules;
- exact final output format;
- a valid example;
- instruction that no `PATH` is allowed.

Suggested content:

```text
[Available Skills]

1. PUT_OBJECT_IN_CONTAINER(object=<object>, container=<bin_slot>)
   Use this skill to place one grocery item into one empty bin slot.
   Valid objects: apple, banana, milk, soda_can, bread, cereal.
   Valid containers: bin_front_left, ...

2. WAIT()
   Use this when the other robot should act and you should remain stationary.

[Coordination Rules]

- Each agent must produce exactly one skill call.
- Do not assign the same object to two agents.
- Do not assign the same bin slot to two agents.
- Do not target an occupied bin slot.
- If an agent holds an object, it may place only that object.
- Do not output PICK, PLACE, MOVE, or PATH.
- The final executable answer must contain one line for Alice and one for Bob.

[Action Output Instruction]

EXECUTE
NAME Alice ACTION <SKILL_CALL>
NAME Bob ACTION <SKILL_CALL>
```

The existing scene description may still include coordinates because they help allocation, but skill mode must state that low-level paths are generated by the executor.

---

# 10. Semantic Skill Validation

Implement a generic `SkillPlanValidator` protocol or base class and a `PackGrocerySkillPlanValidator`.

Suggested interface:

```python
class SkillPlanValidator:
    def validate(self, plan: SkillPlan, obs: EnvState) -> SkillValidationResult:
        raise NotImplementedError
```

## 10.1 Grocery checks

For every plan, validate:

1. All configured agents are present once.
2. Every skill is supported by the assigned agent.
3. `object` is in `env.item_names`.
4. `container` is in `env.bin_slot_xposes`.
5. The object exists in the current observation.
6. The target container/slot exists.
7. The object is not already packed.
8. The target slot is not occupied.
9. The object is not held by another agent.
10. The acting agent is not holding a different object.
11. Two agents do not claim the same object.
12. Two agents do not claim the same target slot.
13. A plan with all agents waiting is rejected to avoid a no-progress loop.
14. `WAIT()` has no arguments.
15. At least one skill can make progress from the current state.

## 10.2 Recovery behavior

Support this state:

```text
Alice already holds apple
Alice calls PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

That is valid and should compile to a place-only legacy action.

Reject this state:

```text
Alice holds banana
Alice calls PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

Reject this state:

```text
Bob holds apple
Alice calls PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

## 10.3 Occupancy detection

Use existing task observations, contact information, and task helpers where possible.

Do not invent a second inconsistent definition of “inside the bin.” If a reusable helper does not exist, add a small task-level helper to `PackGroceryTask`, for example:

```python
def get_packed_slot_for_object(self, obs, object_name) -> Optional[str]: ...
def get_slot_occupancy(self, obs) -> Dict[str, Optional[str]]: ...
def get_agent_held_object(self, obs, agent_name) -> Optional[str]: ...
```

Reuse these methods in:

- semantic validation;
- skill compilation;
- tests;
- future success detection.

Do not mutate simulator state during validation.

---

# 11. RRT Skill Compiler

Implement:

```python
class SkillCompiler:
    def compile(
        self,
        plan: SkillPlan,
        obs: EnvState,
    ) -> PreparedSkillExecution:
        raise NotImplementedError
```

Implement `RRTSkillCompiler` by reusing the existing action-only parser.

## 11.1 Compilation rules

### Agent does not hold the object

```text
PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

becomes:

```text
NAME Alice ACTION PICK apple PLACE bin_front_left
```

### Agent already holds the same object

It becomes:

```text
NAME Alice ACTION PLACE apple bin_front_left
```

### Wait

```text
WAIT()
```

becomes:

```text
NAME Bob ACTION WAIT
```

The final synthetic legacy response should be:

```text
EXECUTE
NAME Alice ACTION PICK apple PLACE bin_front_left
NAME Bob ACTION WAIT
```

Then call an internally configured existing `LLMResponseParser` in `action_only` mode.

## 11.2 Compiler requirements

1. Do not duplicate Cartesian target creation.
2. Do not duplicate grasp-site logic.
3. Do not duplicate plan synchronization.
4. Do not duplicate IK or collision checks.
5. Treat legacy-parser failure as a typed compilation error.
6. Include the synthetic response in diagnostic metadata.
7. Preserve the original skill plan and plan ID.
8. Produce:
   ```python
   PreparedSkillExecution(
       backend_name="rrt",
       source_plan_id=plan.plan_id,
       compiled_plans=[...],
       metadata={"synthetic_response": "..."},
   )
   ```
9. The compiler must be deterministic for the same observation and plan.
10. The compiler should receive dependencies through its constructor, including the legacy parser.

---

# 12. Skill Feedback Adapter

Implement a `SkillFeedbackManager` with the same method expected by existing prompters:

```python
give_feedback(skill_plan) -> Tuple[bool, str]
```

Its flow must be:

```text
semantic validation
    -> if invalid: return actionable feedback

RRT compilation
    -> if compilation fails: return actionable feedback

existing geometric FeedbackManager for every compiled LLMPathPlan
    -> if invalid: return existing geometric feedback

attach PreparedSkillExecution to SkillPlan
    -> return ready
```

Requirements:

- Preserve existing environment, reachability, IK, collision, and waypoint feedback.
- Do not swallow the original geometric feedback.
- Prefix skill-level context where useful:
  ```text
  Skill plan could not be prepared:
  [TARGET_OCCUPIED] ...
  ```
- Never mark a plan ready unless `prepared_execution` is present.
- Clear stale preparation before revalidating a modified plan.
- The feedback manager should hold the current `obs` in the same manner as the legacy manager or expose a clearly documented update method.
- Add unit tests with mocked legacy feedback outcomes.

---

# 13. Skill Executor Interface

Define an abstract interface:

```python
class SkillExecutor(ABC):
    @abstractmethod
    def execute(
        self,
        plan: SkillPlan,
        obs: EnvState,
        artifact_dir: Optional[str] = None,
    ) -> SkillExecutionResult:
        pass
```

The interface must not mention LeRobot or RRT in the generic method signature.

## 13.1 `RRTSkillExecutor`

Implement an executor that:

1. Requires `plan.prepared_execution`.
2. Requires backend name `rrt`.
3. Iterates through compiled `LLMPathPlan` objects.
4. For each:
   - creates `PlannedPathPolicy`;
   - calls `policy.plan(env)`;
   - records planning failure reasons;
   - saves RRT plans and action buffers when an artifact directory is provided;
   - repeatedly calls `policy.act(obs, env.physics)`;
   - calls `env.step(...)`;
   - stops on exhaustion, failure, task completion, or configured safety limit.
5. Returns a structured `SkillExecutionResult`.
6. Does not silently rewind the environment; clearly define whether rollback is owned by the runner or executor.

Recommended ownership:

- `RRTSkillExecutor` executes and reports.
- `LLMRunner` preserves its current transaction behavior:
  - save simulator state before execution;
  - call executor;
  - rewind on failure.

This keeps rollback behavior consistent across future backends.

## 13.2 Dependency injection

Allow a `policy_factory` dependency:

```python
policy_factory: Callable[..., PlannedPathPolicy]
```

This is needed for unit tests and avoids monkeypatching module globals.

## 13.3 Safety limit

Add a configurable maximum number of low-level simulation steps per skill round, with a conservative default or derived bound. Return `TIMEOUT` rather than looping indefinitely.

---

# 14. Runner Integration

## 14.1 CLI

Change:

```python
choices=["action_only", "action_and_path"]
```

to:

```python
choices=["action_only", "action_and_path", "skill"]
```

For Phase 1:

- `--output_mode skill --task pack` is supported.
- Any other task with skill mode should fail early with a clear message:
  ```text
  Skill mode is currently implemented only for --task pack.
  ```

Do not silently fall back to a legacy mode.

## 14.2 Initialization

When `llm_output_mode == "skill"`:

1. Keep response keywords `NAME` and `ACTION`.
2. Do not append `PATH`.
3. Instantiate the normal `MultiArmRRT` because it remains the backend.
4. Build the grocery skill registry.
5. Create the skill prompt provider.
6. Create `SkillResponseParser`.
7. Create an internal legacy action-only `LLMResponseParser` for compilation.
8. Create the existing geometric `FeedbackManager`.
9. Create `PackGrocerySkillPlanValidator`.
10. Create `RRTSkillCompiler`.
11. Create `SkillFeedbackManager`.
12. Create `RRTSkillExecutor`.
13. Pass the skill parser, skill feedback manager, and skill prompt provider to the selected prompter.

When not in skill mode, preserve current construction and behavior.

## 14.3 Execution path

In `one_run(...)`, add a clearly separated skill branch.

Pseudo-flow:

```python
if self.llm_output_mode == "skill":
    skill_plan = current_llm_plan[0]

    save skill_plan.json
    save prepared compilation metadata

    result = self.skill_executor.execute(
        skill_plan,
        obs,
        artifact_dir=step_dir,
    )

    if not result.success:
        rewind_env = True
        env.load_saved_state(sim_data)
    else:
        obs = result.metadata or executor-returned final observation
        reward = result.reward
        done = result.done

    save skill_execution_result.json
else:
    # existing legacy behavior
```

Do not force `SkillPlan` through `display_plan(...)`. Either:

- skip visualization in skill mode; or
- visualize each compiled `LLMPathPlan` after preparation.

The simplest Phase 1 behavior is to skip direct skill visualization and retain existing execution video.

## 14.4 Logging and artifacts

For each skill-mode step, save:

```text
step_N/
├── skill_plan.json
├── skill_preparation.json
├── compiled_path_plans.pkl
├── rrt_plan_0.pkl
├── actions_0.pkl
├── ...
├── skill_execution_result.json
└── execute.mp4
```

JSON files must contain only serializable values.

Suggested `skill_preparation.json`:

```json
{
  "plan_id": "...",
  "backend": "rrt",
  "synthetic_response": "...",
  "compiled_plan_count": 2
}
```

Suggested result:

```json
{
  "success": true,
  "status": "success",
  "reason": "",
  "num_sim_steps": 143,
  "reward": 0,
  "done": false
}
```

Use `logging` in new modules. Avoid new `print()` calls except in the existing CLI-facing runner style.

## 14.5 History and feedback

Ensure:

```python
skill_plan.get_action_desp()
```

is used in:

- replan feedback files;
- round history;
- `post_execute_update(...)`;
- episode HTML.

Do not put synthetic `PICK/PLACE` text into high-level LLM history unless it is explicitly labeled as backend diagnostics.

---

# 15. Backward Compatibility

The implementation is incomplete if legacy behavior changes.

Required regression guarantees:

1. `--output_mode action_only` uses the old parser.
2. `--output_mode action_and_path` uses the old parser.
3. Existing action prompts are unchanged in legacy modes.
4. Existing path instructions remain unchanged in `action_and_path`.
5. Existing `FeedbackManager` behavior is unchanged.
6. Existing `PlannedPathPolicy` is not rewritten.
7. Existing task files other than the minimum required grocery helpers are unchanged.
8. Existing saved artifact names in legacy mode remain available.
9. Existing CLI arguments continue to work.
10. Existing continuation logic does not crash due to new type assumptions.

Prefer small conditional branches and injected interfaces over global type changes.

---

# 16. Testing Requirements

## 16.1 Development dependency

Create `requirements-dev.txt` with a Python 3.8-compatible pytest version, for example:

```text
-r requirements.txt
pytest==7.4.4
```

Do not add production dependencies merely to parse the skill grammar.

## 16.2 Unit tests

### Models

- canonical serialization;
- deterministic plan ID;
- runtime preparation excluded from serialization;
- canonical action description.

### Registry

- registration;
- aliases;
- duplicate names;
- alias collisions;
- unsupported agent;
- missing required arguments;
- unknown arguments;
- deterministic rendered prompt.

### Parser

Test:

- valid coordinated plan;
- agents listed in reverse order;
- discussion before `EXECUTE`;
- missing `EXECUTE`;
- missing agent;
- duplicate agent;
- unknown agent;
- unknown skill;
- missing argument;
- unknown argument;
- duplicate argument;
- positional arguments;
- malformed parentheses;
- `WAIT` with arguments;
- unsafe strings do not execute code.

### Semantic validator

Use small fakes for environment and observation where possible.

Test:

- valid put + wait;
- valid two-agent plan with distinct objects and slots;
- duplicate object conflict;
- duplicate target conflict;
- unknown object;
- unknown target;
- object already packed;
- target occupied;
- object held by another agent;
- agent holding another object;
- valid place-only recovery when agent holds requested object;
- all-wait rejection.

### Compiler

Use a fake or spy legacy parser.

Verify exact synthetic output:

```text
EXECUTE
NAME Alice ACTION PICK apple PLACE bin_front_left
NAME Bob ACTION WAIT
```

Verify place-only output when already held.

Verify compiler propagates legacy-parser errors as typed errors.

### Skill feedback manager

Mock:

- semantic validation failure;
- compilation failure;
- geometric feedback failure;
- fully valid preparation;
- prepared execution attached only on success.

### Executor

Inject a fake policy factory and fake environment.

Test:

- missing prepared execution;
- wrong backend;
- motion-planning failure;
- successful action-buffer execution;
- timeout;
- task done before all plans;
- artifact writing;
- result status and step count.

### Legacy regression

Add at least a small set of existing-format responses to confirm `LLMResponseParser` still parses:

```text
EXECUTE
NAME Alice ACTION PICK apple
NAME Bob ACTION WAIT
```

and, where configured:

```text
EXECUTE
NAME Alice ACTION PICK apple PATH [...]
NAME Bob ACTION WAIT PATH [...]
```

Do not rewrite the old parser tests into skill tests.

## 16.3 Integration test

Mark simulation-heavy tests:

```python
@pytest.mark.integration
```

The integration test must not call an external LLM.

Minimum integration flow:

```text
instantiate PackGroceryTask
    -> obtain observation
    -> parse canned skill response
    -> semantically validate
    -> compile through existing parser
    -> run geometric feedback
    -> assert prepared compiled plans exist
```

An optional slower test may execute one RRT-backed skill if the simulator/assets are available.

## 16.4 Smoke script

Create:

```bash
python scripts/smoke_test_pack_skill.py
```

It must:

1. use a canned skill response;
2. require no OpenAI or Anthropic key;
3. print the parsed canonical skill plan;
4. print the synthetic legacy plan;
5. report semantic and geometric validation results;
6. optionally execute when passed:
   ```bash
   --execute
   ```
7. exit nonzero on failure.

Avoid importing modules that assert the presence of API-key files during import.

---

# 17. Documentation Requirements

Create `docs/phase1_skill_architecture.md` containing:

1. before/after architecture diagrams;
2. exact skill grammar;
3. currently available grocery skills;
4. how validation works;
5. how compilation reuses the legacy parser;
6. how to run tests;
7. how to run the smoke test;
8. how to run a skill-mode dialogue;
9. how to add a new skill;
10. how a future `LearnedSkillExecutor` should plug in;
11. current limitations.

Example command, adjusted to actual repository requirements:

```bash
python run_dialog.py \
  --task pack \
  --output_mode skill \
  --comm_mode dialog \
  --run_name pack_skill_phase1
```

Document that this still uses RRT internally.

---

# 18. Code Quality Requirements

- Python 3.8 compatible.
- Public classes and methods have docstrings.
- New logic has useful type hints.
- No `eval` or `exec`.
- No mutable default arguments.
- No broad `except Exception: pass`.
- No hidden global registry.
- No backend-specific logic in generic models.
- No duplicate IK, collision, grasp, or path code.
- No direct simulator mutation in parser or semantic validator.
- No unnecessary new package dependencies.
- No unrelated formatting churn.
- Use deterministic ordering and stable error messages.
- Use dependency injection for compiler parser and executor policy factory.
- Prefer cohesive, small modules.
- Keep errors suitable for both logs and LLM replanning feedback.
- Ensure new JSON artifact writes use context managers.
- Include plan IDs in logs.
- Keep legacy code path easy to diff and review.

---

# 19. Acceptance Criteria

Phase 1 is complete only when all of the following are true.

## Functional

- [ ] CLI accepts `--output_mode skill`.
- [ ] Skill mode fails early for unsupported tasks.
- [ ] `PackGroceryTask` exposes `PUT_OBJECT_IN_CONTAINER` and `WAIT`.
- [ ] A valid skill response produces a typed `SkillPlan`.
- [ ] The plan is semantically validated.
- [ ] The plan compiles to existing `LLMPathPlan` objects.
- [ ] Existing geometric feedback is executed.
- [ ] The prepared plan executes through `PlannedPathPolicy` and `MultiArmRRT`.
- [ ] Execution failure triggers the existing environment rollback behavior.
- [ ] Skill history remains at the skill abstraction level.
- [ ] Skill and backend artifacts are saved.

## Compatibility

- [ ] `action_only` still works.
- [ ] `action_and_path` still works.
- [ ] Legacy prompts do not change.
- [ ] No LeRobot dependency was added.
- [ ] Python 3.8 compatibility is maintained.

## Quality

- [ ] Unit tests pass.
- [ ] Non-integration tests require no MuJoCo rendering and no API key.
- [ ] Smoke test requires no LLM key.
- [ ] Integration test is clearly marked.
- [ ] Documentation is complete.
- [ ] Codex reviews its own diff for regressions and reports remaining risks.

---

# 20. Required Implementation Order

Codex should implement in this order:

1. Inspect repository and write a concise execution plan.
2. Add models and registry.
3. Add parser and parser tests.
4. Add grocery task-state helpers.
5. Add semantic validator and tests.
6. Add RRT compiler and tests.
7. Add skill feedback adapter and tests.
8. Add skill executor and tests.
9. Add prompt provider.
10. Integrate into prompters.
11. Integrate into runner and CLI.
12. Add smoke script.
13. Add integration test.
14. Add documentation.
15. Run all available checks.
16. Review the final diff for backward compatibility.
17. Report exact commands run, results, and any environmental limitations.

Do not start with `run_dialog.py`; establish and test the leaf abstractions first.

---

# 21. Copy-Ready Master Prompt for Codex

Paste the following into Codex from the repository root.

---

## Codex Prompt

You are implementing **Phase 1: skill-level planning with the existing RRT backend** in this repository.

### Goal

Add a typed skill-planning and skill-execution layer to RoCo for `PackGroceryTask`. The LLM should output one structured skill call per agent, for example:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

The skill plan must remain a first-class `SkillPlan` through validation and runner execution. For this phase, compile valid skill calls into the repository’s existing action-only `LLMPathPlan` representation and execute them with the existing `PlannedPathPolicy` and `MultiArmRRT`.

This phase must prepare the codebase for a future `LearnedSkillExecutor`, but it must not add LeRobot or any learned policy now.

### First action

Before editing code:

1. Read `PHASE1_CODEX_IMPLEMENTATION_SPEC.md` in full.
2. Inspect:
   - `run_dialog.py`
   - `prompting/parser.py`
   - `prompting/feedback.py`
   - `prompting/dialog_prompter.py`
   - `prompting/single_thread_prompter.py`
   - `rocobench/subtask_plan.py`
   - `rocobench/policy.py`
   - `rocobench/envs/base_env.py`
   - `rocobench/envs/task_pack.py`
   - package `__init__.py` files
   - dependency files.
3. Produce a concise implementation plan identifying exact files to add/change, key interfaces, test strategy, and backward-compatibility risks.
4. Then implement the complete plan. Do not stop after writing the plan.

### Required scope

Implement:

- `--output_mode skill`;
- skill mode only for `--task pack`;
- typed models:
  - `SkillSpec`
  - `SkillCall`
  - `SkillPlan`
  - `ValidationIssue`
  - `SkillValidationResult`
  - `PreparedSkillExecution`
  - `SkillExecutionStatus`
  - `SkillExecutionResult`;
- a `SkillRegistry`;
- grocery skills:
  - `PUT_OBJECT_IN_CONTAINER(object, container)`
  - `WAIT()`;
- strict `SkillResponseParser`;
- grocery semantic validation;
- reusable grocery helpers for held object and bin-slot occupancy;
- `RRTSkillCompiler` that creates a synthetic legacy response and reuses the existing action-only `LLMResponseParser`;
- `SkillFeedbackManager` that performs:
  1. semantic validation,
  2. compilation,
  3. existing geometric feedback,
  4. attachment of prepared execution;
- generic `SkillExecutor`;
- `RRTSkillExecutor` using existing `PlannedPathPolicy`;
- an injectable skill prompt provider so skill mode contains no `PATH` instructions;
- runner integration, artifacts, rollback, tests, smoke script, and documentation.

### Exact compilation behavior

If the acting agent is not holding the object:

```text
PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

must compile to:

```text
PICK apple PLACE bin_front_left
```

If the acting agent already holds `apple`, compile to:

```text
PLACE apple bin_front_left
```

`WAIT()` compiles to:

```text
WAIT
```

Use the existing parser to convert the complete synthetic response into synchronized `LLMPathPlan` objects. Do not recreate Cartesian targets, grasp logic, synchronization, IK, collision checks, or RRT logic.

### Semantic validation requirements

Reject with stable, actionable error codes:

- unknown skill;
- unsupported agent;
- unknown object;
- unknown bin slot;
- object already packed;
- target occupied;
- object held by another agent;
- agent holding a different object;
- duplicate object claim;
- duplicate target claim;
- all agents waiting;
- malformed or missing arguments.

Support place-only recovery when the assigned agent already holds the requested object.

### Parser requirements

- Require `EXECUTE`.
- Require exactly one call for every configured agent.
- Reject duplicate or unknown agents.
- Parse only canonical function-call syntax.
- Reject positional arguments.
- Reject unknown or duplicate arguments.
- Resolve skill names case-insensitively but render canonical names.
- Never use `eval` or `exec`.
- Return expected formatting failures through the existing `(success, message, plans)` pattern.
- Return a list containing one coordinated `SkillPlan`.
- Implement `SkillPlan.get_action_desp()` so existing history and logging code can use it.

### Backward compatibility

Do not alter behavior for:

```text
--output_mode action_only
--output_mode action_and_path
```

Legacy prompts, parser behavior, feedback, execution, artifacts, and CLI arguments must continue to work.

Use explicit skill-mode branches or injected interfaces. Avoid a broad rewrite of stable legacy code.

### Prompting

The current grocery prompt mentions paths. In skill mode, provide a skill-specific prompt containing:

- current scene and robot holding state;
- valid object names and bin-slot names;
- per-agent skill list;
- exact argument names;
- resource-conflict rules;
- exact `EXECUTE / NAME / ACTION` format;
- instruction not to output `PICK`, `PLACE`, `MOVE`, or `PATH`.

Modify applicable prompters to accept an optional prompt provider, defaulting to existing environment methods in legacy modes.

### Execution and rollback

The generic executor method must not mention RRT or LeRobot.

`RRTSkillExecutor` must:

- require prepared RRT execution;
- iterate through compiled `LLMPathPlan` objects;
- create and run `PlannedPathPolicy`;
- return a structured result;
- support a maximum low-level step limit;
- accept a policy factory for unit testing;
- save RRT/action artifacts when an artifact directory is supplied.

Keep rollback ownership in `LLMRunner`: save state before execution and restore it if the executor reports failure.

### Artifacts

In skill mode save:

```text
skill_plan.json
skill_preparation.json
compiled_path_plans.pkl
rrt_plan_<i>.pkl
actions_<i>.pkl
skill_execution_result.json
execute.mp4
```

Keep high-level history in skill terms. The synthetic legacy response is backend diagnostic metadata, not the user-facing plan.

### Testing

Add Python 3.8-compatible pytest tests for:

- models;
- registry;
- parser success and failures;
- grocery semantic validation;
- resource conflicts;
- place-only recovery;
- compiler’s exact synthetic response;
- feedback adapter;
- executor outcomes and timeout;
- legacy parser regression.

Add an integration test marked `integration` that uses no external LLM.

Add:

```bash
python scripts/smoke_test_pack_skill.py
```

It must use a canned response and no API key. It should parse, validate, compile, and print results; `--execute` may run the RRT backend.

Avoid importing any module that requires an API-key file during test collection.

### Quality constraints

- Python 3.8 syntax.
- No new production dependency unless absolutely necessary.
- No mutable default arguments.
- No unsafe parsing.
- No hidden global registry.
- No broad exception swallowing.
- No unrelated refactors.
- Public APIs have docstrings and useful type hints.
- Use dependency injection for testability.
- Use deterministic ordering and stable error codes.
- Prefer context managers for files.
- Use logging in new modules.
- Preserve import boundaries and avoid circular dependencies.

### Done when

Do not declare completion until:

1. all non-integration tests pass;
2. the smoke test passes without an LLM key;
3. `python -m compileall` passes on changed Python modules;
4. legacy parser regression tests pass;
5. the diff has been self-reviewed for behavior changes in legacy modes;
6. documentation explains architecture, grammar, extension process, commands, and limitations;
7. you provide a final report listing:
   - files added/changed;
   - architecture implemented;
   - tests/commands run and results;
   - any tests not runnable in the current environment;
   - remaining risks or follow-up work.

If simulator dependencies are unavailable, complete and run all non-simulator tests, clearly report the blocked integration command, and do not pretend it passed.

---

# 22. Suggested Repository `AGENTS.md`

Codex automatically reads repository guidance. Add or adapt this at the repository root before the implementation session if the repository does not already have stronger instructions.

```markdown
# AGENTS.md

## Repository purpose

This repository implements RoCo, a language-based multi-robot coordination framework using MuJoCo, task-specific prompting, geometric validation, inverse kinematics, and multi-arm RRT execution.

## Runtime constraints

- Preserve Python 3.8 compatibility.
- The existing environment depends on MuJoCo 2.3-era packages.
- Do not introduce LeRobot or Python 3.12 dependencies in Phase 1.
- Avoid changing production dependencies without explicit justification.

## Architecture rules

- Keep language planning, task-semantic validation, backend compilation, and low-level execution separated.
- Reuse existing `LLMResponseParser`, `FeedbackManager`, `PlannedPathPolicy`, and `MultiArmRRT`.
- Do not duplicate IK, collision, grasp, waypoint, or RRT logic.
- New skill abstractions must remain backend-independent.
- Preserve the legacy `action_only` and `action_and_path` paths.
- Task-specific skill rules belong outside generic skill models.
- Avoid circular imports between `prompting`, `rocobench.skills`, policies, and environments.

## Coding rules

- Use Python 3.8-compatible typing.
- Do not use `eval` or `exec`.
- Do not use mutable default arguments.
- Use dataclasses for new domain models.
- Provide deterministic serialization and stable validation error codes.
- Use dependency injection for parser/compiler/executor collaborators.
- Use context managers for file I/O.
- Use logging in new modules.
- Do not make unrelated formatting changes.

## Verification

For Phase 1, run:

```bash
python -m pytest -m "not integration"
python scripts/smoke_test_pack_skill.py
python -m compileall rocobench/skills prompting
```

Run simulator integration tests when dependencies and assets are available:

```bash
python -m pytest -m integration
```

## Definition of done

- Requested behavior is implemented.
- Legacy modes are covered by regression tests.
- New code has unit tests.
- No external LLM is needed for unit/smoke tests.
- Documentation and commands are updated.
- Final response reports exact checks run and any blocked checks.
```

---

# 23. Follow-Up Review Prompt for Codex

After Codex implements the changes, run a separate review turn:

```text
Review the complete Phase 1 diff as a senior robotics software engineer.

Focus on:
1. hidden regressions in action_only and action_and_path modes;
2. violations of the skill/backend separation;
3. parser safety and malformed-input handling;
4. inconsistent definitions of held objects or occupied bin slots;
5. stale or missing prepared execution state;
6. rollback behavior after partial execution;
7. import cycles and API-key-dependent imports during tests;
8. Python 3.8 incompatibilities;
9. nondeterministic plan ordering or IDs;
10. missing tests for failure paths.

Do not edit first. Produce findings ordered by severity with file and line references. Then fix all confirmed issues, run relevant tests, and summarize the fixes and results.
```

---

# 24. Follow-Up Verification Prompt for Codex

Use this after review:

```text
Verify Phase 1 end to end without using an external LLM.

1. Run all non-integration tests.
2. Run the canned grocery skill smoke test.
3. Run compileall on all changed Python packages.
4. Exercise parser failure cases for:
   - duplicate agent,
   - unknown skill,
   - missing argument,
   - duplicate object,
   - duplicate target,
   - all WAIT.
5. Confirm a valid plan compiles to the exact expected legacy response.
6. Confirm action_only and action_and_path still select the legacy parser and prompts.
7. If MuJoCo is available, run the marked integration test and one `--execute` smoke case.
8. Inspect generated JSON artifacts for serializability and useful content.
9. Report commands, exit codes, passed/failed/skipped counts, and any remaining blocker.

Do not claim a simulator test passed unless it actually ran.
```

---

# 25. Expected Phase 1 End-State

After implementation, the code should conceptually support:

```python
skill_plan = skill_parser.parse(obs, llm_response)

ready, feedback = skill_feedback_manager.give_feedback(skill_plan)

if ready:
    result = skill_executor.execute(skill_plan, obs, artifact_dir=step_dir)
```

With Phase 1 binding:

```python
skill_executor = RRTSkillExecutor(...)
```

And a future phase able to bind:

```python
skill_executor = LearnedSkillExecutor(...)
```

without changing:

- the LLM output grammar;
- `SkillCall`;
- `SkillPlan`;
- registry semantics;
- task-level resource validation;
- prompt history;
- the high-level coordination protocol.
