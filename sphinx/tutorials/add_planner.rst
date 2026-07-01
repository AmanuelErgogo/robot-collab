Tutorial: Adding a New Planner
================================

A planner converts a task goal + observation (+ optional failure feedback)
into a :class:`~rocobench.crie_bt.CollaborativePlan`. Any object that
implements :class:`~rocobench.crie_bt.planner.BasePlanner` works as a
drop-in replacement.

This tutorial shows three ways to add a planner:

1. Custom rule-based planner (no LLM).
2. Wrap an existing LLM prompter (Centralised or Dialog).
3. Implement a fully custom LLM planner.

Option A — Rule-based planner
-------------------------------

Subclass :class:`~rocobench.crie_bt.planner.BasePlanner` and implement
``generate_plan``:

.. code-block:: python

   from rocobench.crie_bt.planner import BasePlanner
   from rocobench.crie_bt.types import (
       CollaborativePlan, PlanStep, SkillCall,
   )

   class FixedOrderSandwichPlanner(BasePlanner):
       """Always plans the next recipe step based on observation state."""

       RECIPE = ["bread_slice1", "bacon", "cheese", "tomato", "bread_slice2"]

       def generate_plan(self, task_goal, observation, feedback=None, context=None):
           # Determine which item comes next from the observation
           placed = getattr(observation, "placed_items", [])
           for item in self.RECIPE:
               if item not in placed:
                   next_item = item
                   break
           else:
               next_item = self.RECIPE[-1]   # already done

           call = SkillCall(
               agent="Dave",
               skill_name="PICK",
               arguments={"object": next_item},
           )
           step = PlanStep(
               step_id="step_001",
               skill_call=call,
               explanation="Pick {} next in the recipe.".format(next_item),
           )
           return CollaborativePlan(
               plan_id="fixed_plan_001",
               task_goal=task_goal,
               steps=[step],
           )

Option B — Wrap a RoCoBench LLM prompter (Centralised / Dialog)
-----------------------------------------------------------------

:class:`~rocobench.crie_bt.LegacyPromptPlanner` wraps any RoCoBench
``SingleThreadPrompter`` (for Centralised mode) or ``DialogPrompter``
(for Dialog mode) behind the ``BasePlanner`` interface.

**Centralised with history (chat mode)**

.. code-block:: python

   from rocobench.crie_bt import LegacyPromptPlanner
   from prompting import SingleThreadPrompter   # RoCoBench prompter

   prompter = SingleThreadPrompter(
       agent_names=["Chad", "Dave"],
       env=env,
       llm_source="gemini",
       ...
   )
   planner = LegacyPromptPlanner(
       prompter=prompter,
       task_id="sandwich",
       planner_mode="chat",    # "plan" (no history) | "chat" | "dialog"
       num_replans=2,
   )

**Dialog mode**

.. code-block:: python

   from prompting import DialogPrompter

   prompter = DialogPrompter(
       agent_names=["Chad", "Dave"],
       env=env,
       llm_source="gemini",
       ...
   )
   planner = LegacyPromptPlanner(
       prompter=prompter,
       task_id="sandwich",
       planner_mode="dialog",
       num_replans=2,
   )

The ``planner_mode`` controls how feedback from previous steps is threaded
into the next prompt (see :doc:`/guides/concepts`).

Option C — Fully custom LLM planner
-------------------------------------

If you have your own LLM client, implement ``generate_plan`` directly:

.. code-block:: python

   import json
   from rocobench.crie_bt.planner import BasePlanner
   from rocobench.crie_bt.types import CollaborativePlan, PlanStep, SkillCall

   class MyGPT4Planner(BasePlanner):

       def __init__(self, client, model="gpt-4o"):
           self.client = client
           self.model = model

       def generate_plan(self, task_goal, observation, feedback=None, context=None):
           prompt = self._build_prompt(task_goal, observation, feedback)
           response = self.client.chat.completions.create(
               model=self.model,
               messages=[{"role": "user", "content": prompt}],
           )
           return self._parse_response(response.choices[0].message.content, task_goal)

       def _build_prompt(self, task_goal, observation, feedback):
           lines = ["Task: {}".format(task_goal),
                    "Observation: {}".format(observation)]
           if feedback and feedback.failure.is_failure:
               lines.append("Last failure: {}".format(feedback.failure.failure_code.value))
           lines.append("Reply with: AGENT ACTION OBJECT")
           return "\n".join(lines)

       def _parse_response(self, text, task_goal):
           # parse "Dave PICK bread_slice1" style response
           parts = text.strip().split()
           call = SkillCall(agent=parts[0], skill_name=parts[1],
                            arguments={"object": parts[2]})
           step = PlanStep(step_id="step_001", skill_call=call)
           return CollaborativePlan(steps=[step], task_goal=task_goal)

Using the planner
------------------

All three planner types are interchangeable:

.. code-block:: python

   from rocobench.crie_bt import OpenLoopController, ScriptedSkillExecutor

   controller = OpenLoopController(
       planner=planner,   # any BasePlanner subclass
       executor=ScriptedSkillExecutor(),
   )
   log = controller.run_episode(env=env, task_goal="assemble sandwich", max_steps=20)
