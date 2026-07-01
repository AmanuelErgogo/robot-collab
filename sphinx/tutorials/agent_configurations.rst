Tutorial: Agent Configurations
================================

CRIE-Bench supports four team compositions. The key design principle is that
**all configurations share the same execution pipeline** — the only difference
is who (or what) chooses the next subtask.

Robot–Robot
-----------

Both agents are planned by an LLM. This is the primary configuration for all
six experimental conditions.

.. code-block:: python

   from rocobench.crie_bt import (
       BTMediatedController,
       LegacyPromptPlanner,
       LegacyTaskRRTExecutorAdapter,
   )

   planner = LegacyPromptPlanner(
       prompter=my_llm_prompter,   # SingleThreadPrompter or DialogPrompter
       task_id="sandwich",
       planner_mode="chat",
   )
   executor = LegacyTaskRRTExecutorAdapter(task_id="sandwich", env=env)
   controller = BTMediatedController(planner=planner, executor=executor)

   log = controller.run_episode(env=env, task_goal="assemble sandwich", max_steps=50)

Human–Robot
-----------

One agent is a human operator; the other is LLM-planned. The human's subtask
selection is captured via a simple callable:

.. code-block:: python

   from rocobench.crie_bt.planner import BasePlanner
   from rocobench.crie_bt.types import CollaborativePlan, PlanStep, SkillCall

   class HumanOperatorPlanner(BasePlanner):
       """Reads a subtask selection from stdin (or a GUI widget)."""

       def __init__(self, agent_name: str):
           self.agent_name = agent_name

       def generate_plan(self, task_goal, observation, feedback=None, context=None):
           print("\nObservation:", observation)
           print("Enter action (e.g. PICK bread_slice1): ", end="")
           raw = input().strip()
           parts = raw.split()
           call = SkillCall(
               agent=self.agent_name,
               skill_name=parts[0],
               arguments={"object": parts[1]} if len(parts) > 1 else {},
           )
           step = PlanStep(step_id="human_step", skill_call=call)
           return CollaborativePlan(steps=[step], task_goal=task_goal)

   human_planner = HumanOperatorPlanner(agent_name="Dave")
   llm_planner   = LegacyPromptPlanner(prompter=my_prompter, task_id="sandwich",
                                        planner_mode="chat")

   # Combine into a joint planner that dispatches by agent
   class JointHumanRobotPlanner(BasePlanner):
       def __init__(self, human, robot):
           self.human = human
           self.robot = robot

       def generate_plan(self, task_goal, observation, feedback=None, context=None):
           # The LLM plans for Chad; the human selects Dave's action
           llm_plan   = self.robot.generate_plan(task_goal, observation, feedback, context)
           human_plan = self.human.generate_plan(task_goal, observation, feedback, context)
           combined_steps = llm_plan.steps + human_plan.steps
           return CollaborativePlan(steps=combined_steps, task_goal=task_goal)

   planner = JointHumanRobotPlanner(human=human_planner, robot=llm_planner)

.. note::

   The human planner and the LLM planner both return a
   :class:`~rocobench.crie_bt.CollaborativePlan` with a single step. The
   executor receives both steps and executes them in order, so collision
   avoidance is handled by the RRT planner exactly as in Robot–Robot mode.

Human–Human (upper-bound baseline)
-----------------------------------

Both agents are controlled by human operators. No LLM is involved.

.. code-block:: python

   human_chad = HumanOperatorPlanner(agent_name="Chad")
   human_dave = HumanOperatorPlanner(agent_name="Dave")

   class BothHumanPlanner(BasePlanner):
       def generate_plan(self, task_goal, observation, feedback=None, context=None):
           chad_plan = human_chad.generate_plan(task_goal, observation, feedback, context)
           dave_plan = human_dave.generate_plan(task_goal, observation, feedback, context)
           return CollaborativePlan(
               steps=chad_plan.steps + dave_plan.steps,
               task_goal=task_goal,
           )

   planner = BothHumanPlanner()
   controller = DirectFeedbackController(planner=planner, executor=executor)

This configuration establishes the performance ceiling. Human–Human results
are used as the reference baseline in all result tables.

Single Robot
------------

Only one arm is active. Pass a planner that generates plans for a single
agent and configure the executor to control only that arm.

.. code-block:: python

   from rocobench.crie_bt import ScriptedPlanner, OpenLoopController

   planner = ScriptedPlanner(default_agent="Chad")   # only Chad acts
   controller = OpenLoopController(planner=planner, executor=single_arm_executor)

   log = controller.run_episode(env, task_goal="pick bread_slice1", max_steps=10)

.. note::

   Single-Robot mode is only feasible for tasks that one arm can physically
   complete. Tightly-Coupled tasks (Rope) and Continuous-Cooperative tasks
   (Sweep) require two arms by design.

Choosing the right configuration
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Goal
     - Configuration
   * - Evaluate LLM planning quality
     - Robot–Robot
   * - Test asymmetric teaming / HRC
     - Human–Robot
   * - Establish performance ceiling
     - Human–Human
   * - Isolate single-agent capability
     - Single Robot
   * - Fast offline ablation (no simulator)
     - Any configuration with ``ScriptedSkillExecutor``
