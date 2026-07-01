Quickstart
==========

This page gets you from zero to a running episode in under five minutes using
only fake components — no MuJoCo, no LLM API key required.

.. code-block:: python

   from rocobench.crie_bt import (
       OpenLoopController,
       ScriptedPlanner,
       ScriptedSkillExecutor,
   )

   # 1. Choose a planner.
   #    ScriptedPlanner is deterministic and needs no LLM.
   planner = ScriptedPlanner(default_agent="Dave")

   # 2. Choose an executor.
   #    ScriptedSkillExecutor simulates execution without a simulator.
   executor = ScriptedSkillExecutor(scenario="none")   # always succeeds

   # 3. Choose a controller (feedback mode).
   controller = OpenLoopController(planner=planner, executor=executor)

   # 4. Run an episode.
   #    env=None is fine for ScriptedSkillExecutor.
   log = controller.run_episode(env=None, task_goal="pick apple", max_steps=10)

   print(log["success"])      # True
   print(log["steps"])        # 1
   print(log["failure_counts"])  # {}

Simulate a failure and recovery
--------------------------------

Switch to :class:`~rocobench.crie_bt.DirectFeedbackController` and inject a
``missed_grasp`` on the first attempt:

.. code-block:: python

   from rocobench.crie_bt import DirectFeedbackController, ScriptedSkillExecutor

   executor = ScriptedSkillExecutor(
       scenario="missed_grasp",
       fail_attempts=1,   # fail once, then succeed
   )
   controller = DirectFeedbackController(
       planner=ScriptedPlanner(),
       executor=executor,
   )
   log = controller.run_episode(env=None, task_goal="pick apple", max_steps=20)

   print(log["success"])     # True  — recovered after replan
   print(log["replans"])     # 1

Run on the sandwich task (MuJoCo)
----------------------------------

If you have the ``roco`` conda environment and MuJoCo installed:

.. code-block:: bash

   MUJOCO_GL=egl conda run -n roco \
       python scripts/run_crie_bt_sim.py \
           --task sandwich \
           --mode open_loop \
           --comm plan \
           --num-episodes 1

See :doc:`/tutorials/run_sandwich` for the full walkthrough.

Next steps
----------

* :doc:`/guides/concepts` — understand the five building blocks.
* :doc:`/tutorials/add_task` — register a new MuJoCo task.
* :doc:`/tutorials/add_skill` — add a new skill to the registry.
* :doc:`/tutorials/add_planner` — plug in your own LLM planner.
