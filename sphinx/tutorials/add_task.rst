Tutorial: Adding a New Task
============================

A CRIE-Bench task is a MuJoCo environment plus a thin adapter that translates
between the simulator's native API and the ``BasePlanner`` / ``BaseSkillExecutor``
contract. You need to write two things:

1. A **task environment** (subclass of the RoCoBench base env, or any object
   with ``reset()`` / ``step()`` / ``get_reward_done()``).
2. A **CRIE-BT adapter** that wires the planner and executor to your env.

This tutorial adds a minimal "pick-and-place" task as an example.

Step 1 — Write the environment
-------------------------------

Create ``rocobench/envs/task_pick_place.py``:

.. code-block:: python

   class PickPlaceEnv:
       """Minimal two-arm pick-and-place task."""

       robot_name_map = {
           "ur5e_robotiq": "Alice",
           "panda": "Bob",
       }

       def reset(self):
           self._done = False
           self._step = 0
           return self._observation()

       def step(self, actions):
           """actions: dict[agent_name -> action_str]"""
           self._step += 1
           if all(a == "PLACE target" for a in actions.values()):
               self._done = True
           return self._observation(), {}, self._done, {}

       def get_reward_done(self, obs=None):
           return 0.0 if not self._done else 1.0, self._done

       def _observation(self):
           return {"step": self._step, "done": self._done}

The environment only needs to satisfy:

* ``reset() -> obs`` — reset and return the initial observation.
* ``get_reward_done(obs) -> (reward, done)`` — used by the controller to
  detect task completion.

Step 2 — Register a task spec
-------------------------------

CRIE-BT controllers look up task metadata through a ``TaskSpec``. Add yours
alongside the existing task specs, for example in
``rocobench/crie_bt/legacy_tasks.py`` or your own module:

.. code-block:: python

   from rocobench.crie_bt.legacy_tasks import TaskSpec, register_task

   pick_place_spec = TaskSpec(
       task_id="pick_place",
       display_name="Pick and Place",
       agents=["Alice", "Bob"],
       coordination_class="parallel_independent",
       max_steps=20,
       description="Alice and Bob each pick one object and place it at the target.",
   )

   register_task(pick_place_spec)

Step 3 — Write the CRIE-BT adapter
-------------------------------------

The adapter is a :class:`~rocobench.crie_bt.planner.BasePlanner` that reads
the environment observation and constructs a :class:`~rocobench.crie_bt.CollaborativePlan`.
For tasks that use the RoCoBench LLM prompters, subclass
:class:`~rocobench.crie_bt.LegacyPromptPlanner`:

.. code-block:: python

   from rocobench.crie_bt import LegacyPromptPlanner

   class PickPlaceLLMPlanner(LegacyPromptPlanner):
       """Wraps a RoCoBench SingleThreadPrompter for the pick-and-place task."""

       def __init__(self, prompter, num_replans=2):
           super().__init__(
               prompter=prompter,
               task_id="pick_place",
               num_replans=num_replans,
           )

For the executor, subclass
:class:`~rocobench.crie_bt.executor.BaseSkillExecutor` and call your env's
``step()``:

.. code-block:: python

   from rocobench.crie_bt import BaseSkillExecutor, BTStatus
   from rocobench.crie_bt.types import ExecutionFeedback, FailureState, ProgressState

   class PickPlaceExecutor(BaseSkillExecutor):

       def reset(self, env, context):
           self.env = env
           self.context = context

       def start_skill(self, skill_call, observation):
           self.skill_call = skill_call

       def step(self, observation):
           _, _, done, _ = self.env.step(
               {self.skill_call.agent: str(self.skill_call)}
           )
           status = BTStatus.SUCCESS if done else BTStatus.RUNNING
           return ExecutionFeedback(
               skill_call=self.skill_call,
               status=status,
               progress=ProgressState(postcondition_satisfied=done),
               failure=FailureState(is_failure=False),
           )

       def stop(self):
           pass

Step 4 — Run an episode
------------------------

.. code-block:: python

   from rocobench.crie_bt import OpenLoopController, ScriptedPlanner
   from rocobench.envs.task_pick_place import PickPlaceEnv

   env = PickPlaceEnv()
   planner = ScriptedPlanner(default_agent="Alice")
   executor = PickPlaceExecutor()
   controller = OpenLoopController(planner=planner, executor=executor)

   log = controller.run_episode(env, task_goal="pick and place", max_steps=10)
   print(log)

Step 5 — Add to the evaluation suite
--------------------------------------

Add your task to ``scripts/eval_sandwich_open_loop.py`` (or a new eval
script) by passing ``task="pick_place"`` to the builder function. See
:doc:`/tutorials/evaluate_and_log` for the full evaluation pipeline.
