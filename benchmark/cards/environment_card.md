# Environment Card: RoCoBench-Pack-Skills-v1

## Scope

This benchmark covers PackGrocery skill execution with Alice and Bob in RoCoBench.

## Tasks

- `pack.put.alice`
- `pack.put.bob`
- `pack.sequential.two_agent`
- `pack.concurrent.safe_two_agent`

The concurrent task is included only because Phase 7 has a passing centralized joint-step smoke artifact.

## Robots And Control

- Alice: UR5e Robotiq in the RoCo simulator.
- Bob: Panda in the RoCo simulator.
- Bridge policy action representation: `absolute_joint_position_plus_gripper`.
- Typed planner action representation: `SkillPlan` with `PUT_OBJECT_IN_CONTAINER` and `WAIT`.

## Observations

The benchmark processor maps raw bridge observations only:

- `agent_pos` to `observation.state`
- `pixels.front` to `observation.images.front`
- `pixels.active_agent` to `observation.images.active_agent`

Normalization, batching, tokenization, and device placement remain policy-processor responsibilities.

## Predicates

Success uses PackGrocery task predicates and postconditions:

- assigned object packed in assigned slot for single-skill tasks;
- all assigned objects packed in assigned slots for multi-agent tasks.

## Variations

Variation seeds, assignments, targets, distractors, and concurrency cases are fixed in `benchmark/manifests/pack_v1_variations.json`. The full manifest hash is recorded in every result.

## Dependency Architecture

The RoCo simulator remains in the Python 3.8 runtime. LeRobot/Gym client code runs in the isolated bridge client runtime and communicates through the Phase 0 protocol.

## Limits

The default ACT debug checkpoint is for pipeline validation and may fail manipulation. Learned+RRT fallback requires an injected Phase 5 executor and is not fabricated by the CLI.

