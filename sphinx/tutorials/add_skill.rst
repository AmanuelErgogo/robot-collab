Tutorial: Adding a New Skill
=============================

A **skill** is the atomic unit of physical execution. It maps to one
:class:`~rocobench.skills.models.SkillSpec` in the
:class:`~rocobench.skills.registry.SkillRegistry` and is executed by a
:class:`~rocobench.crie_bt.executor.BaseSkillExecutor`.

This tutorial registers a ``FLIP`` skill that turns an object upside-down.

What is a SkillSpec?
--------------------

:class:`~rocobench.skills.models.SkillSpec` declares the planner-visible
interface of a skill — its name, required arguments, optional arguments,
which agents can use it, and human-readable aliases.

.. code-block:: python

   from rocobench.skills.models import SkillSpec

   FLIP_SKILL = SkillSpec(
       name="FLIP",
       description="Flip an object upside-down in place.",
       required_arguments=("object",),
       optional_arguments=("speed",),
       supported_agents=("Alice", "Bob"),   # empty tuple = all agents
       aliases=("TURN_OVER", "INVERT"),
   )

Rules:

* ``name`` must be uppercase snake-case (``FLIP``, ``PUT_IN_BOX``).
* ``required_arguments`` must be satisfied by every planner call.
* Aliases let the LLM use alternate names; the registry resolves them to the
  canonical name before execution.
* ``resource_arguments`` names the arguments that represent shared workspace
  resources (used by the safety monitor to detect conflicts).

Step 1 — Register the skill
----------------------------

Add the spec to the shared registry in ``rocobench/skills/registry.py`` or
register it at runtime:

.. code-block:: python

   from rocobench.skills.registry import SkillRegistry
   from rocobench.skills.models import SkillSpec

   registry = SkillRegistry()
   registry.register(FLIP_SKILL)

   # Resolve an alias
   assert registry.resolve_name("TURN_OVER") == "FLIP"

   # Check what Alice can do
   for spec in registry.skills_for_agent("Alice"):
       print(spec.name)

Step 2 — Implement execution
-----------------------------

Add a branch in your executor's ``start_skill`` method:

.. code-block:: python

   def start_skill(self, skill_call, observation):
       self.skill_call = skill_call
       if skill_call.skill_name == "FLIP":
           obj = skill_call.arguments["object"]
           self.env.set_flip_target(obj)
       elif skill_call.skill_name == "PICK":
           ...

For RRT-based execution, compile the skill into a path plan using the
existing ``SkillCompiler`` in ``rocobench/skills/compiler.py``.

Step 3 — Validate a call
-------------------------

:class:`~rocobench.skills.registry.SkillRegistry` validates argument
completeness before execution:

.. code-block:: python

   from rocobench.skills.models import SkillCall

   call = SkillCall(
       agent_name="Alice",
       skill_name="FLIP",
       arguments={"object": "cup"},
       raw_action="FLIP cup",
   )

   result = registry.validate(call)
   if not result.valid:
       print(result.issues)

Step 4 — Expose to the planner
--------------------------------

When building the LLM prompt, the planner enumerates available skills via
``registry.skills_for_agent(agent_name)`` and formats them into the system
prompt. No extra step required — registering the spec is sufficient.

Step 5 — Test without a simulator
----------------------------------

Use :class:`~rocobench.crie_bt.ScriptedSkillExecutor` to test the controller
logic without touching MuJoCo:

.. code-block:: python

   from rocobench.crie_bt import (
       OpenLoopController, ScriptedPlanner, ScriptedSkillExecutor,
   )

   planner = ScriptedPlanner()
   executor = ScriptedSkillExecutor(scenario="none")   # always succeeds
   controller = OpenLoopController(planner=planner, executor=executor)

   log = controller.run_episode(
       env=None,
       task_goal="FLIP object=cup",
       max_steps=5,
   )
   assert log["success"]
