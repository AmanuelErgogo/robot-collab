Core Concepts
=============

This page explains the five building blocks of CRIE-Bench before you look at
any code. Read this once and the rest of the documentation will make sense
immediately.

The five building blocks
------------------------

1. Subtask / SkillCall
~~~~~~~~~~~~~~~~~~~~~~

The atomic unit of work in CRIE-Bench is a **subtask**: one agent performing
one named skill on one or more named objects. It is represented by
:class:`~rocobench.crie_bt.SkillCall`:

.. code-block:: python

   from rocobench.crie_bt import SkillCall

   call = SkillCall(
       agent="Dave",
       skill_name="PICK",
       arguments={"object": "bread_slice1"},
       instruction="Pick the first bread slice.",
   )

``skill_name`` is always normalised to upper-snake-case (``PICK``, ``PLACE``,
``WAIT``). The ``arguments`` dict maps parameter names to object identifiers
that come from the environment's observation.

2. Planner
~~~~~~~~~~

A :class:`~rocobench.crie_bt.planner.BasePlanner` takes the current
**observation** and (optionally) **execution feedback** from the previous step
and returns a :class:`~rocobench.crie_bt.CollaborativePlan` — an ordered list
of :class:`~rocobench.crie_bt.PlanStep` objects, each wrapping one
``SkillCall``.

Three planner types ship out of the box:

* :class:`~rocobench.crie_bt.ScriptedPlanner` — deterministic, for tests and
  offline ablations.
* :class:`~rocobench.crie_bt.LegacyPromptPlanner` — wraps any RoCoBench
  ``SingleThreadPrompter`` or ``DialogPrompter`` (Centralised or Dialog LLM
  communication mode).
* :class:`~rocobench.crie_bt.LLMPlannerAdapter` — thin adapter for custom
  LLM planners.

A *human* planner is not a class — it is a UI that lets an operator pick a
``SkillCall`` from the valid action set. The selected call is passed to the
executor through the same interface.

3. Skill Executor
~~~~~~~~~~~~~~~~~

A :class:`~rocobench.crie_bt.executor.BaseSkillExecutor` receives a
``SkillCall`` and drives the robot through the physical action, returning
:class:`~rocobench.crie_bt.ExecutionFeedback` on every simulator tick.

``ExecutionFeedback`` contains three orthogonal signals:

* **progress** (:class:`~rocobench.crie_bt.ProgressState`) — which stage the
  skill is at (``APPROACHING_OBJECT``, ``GRASPED``, ``TRANSPORTING``,
  ``STABLE_SUCCESS``, ``FAILED``, …).
* **uncertainty** (:class:`~rocobench.crie_bt.UncertaintyState`) — estimated
  confidence in the current execution (``confidence``, ``risk_level``).
* **failure** (:class:`~rocobench.crie_bt.FailureState`) — whether a failure
  was detected and which :class:`~rocobench.crie_bt.FailureCode` applies.

Executor implementations:

* :class:`~rocobench.crie_bt.ScriptedSkillExecutor` — simulates deterministic
  scenarios (``missed_grasp``, ``slippage``, ``no_progress``, …) without a
  simulator. Used in unit tests.
* :class:`~rocobench.crie_bt.RRTSkillExecutor` — delegates to the RoCoBench
  RRT motion planner. Requires a live MuJoCo environment.
* :class:`~rocobench.crie_bt.LearnedSkillExecutor` — delegates to an ACT /
  Diffusion / LeRobot policy loaded from a checkpoint.

4. Controller
~~~~~~~~~~~~~

A **controller** owns the episode loop. It calls the planner, dispatches
subtasks to the executor, reads feedback, and decides what to do next. Three
controllers are provided:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Controller
     - Behaviour
   * - :class:`~rocobench.crie_bt.OpenLoopController`
     - Plans once at the start of the episode; executes all steps in sequence
       with no recovery. Failure of one step halts the episode.
   * - :class:`~rocobench.crie_bt.DirectFeedbackController`
     - After each failed step, sends failure details back to the planner and
       replans. Loops until the task succeeds or the step budget is exhausted.
   * - :class:`~rocobench.crie_bt.BTMediatedController`
     - The BT reads the failure code and uncertainty level and decides whether
       to replan, retry the same skill locally, or escalate to a human
       operator. The proposed method in the CRIE-BT paper.

5. Episode Log
~~~~~~~~~~~~~~

Every call to ``controller.run_episode(env, task_goal, max_steps)`` returns a
plain ``dict`` that is directly JSON-serialisable. Key fields:

.. code-block:: python

   {
       "mode": "open_loop",
       "task_goal": "assemble bacon sandwich",
       "success": True,           # all plan steps succeeded
       "sim_success": False,      # env.get_reward_done() == True
       "steps": 1,
       "planner_calls": 1,
       "replans": 0,
       "failure_counts": {},
       "subtask_results": [...],
       "wall_time_s": 9.3,
   }

The evaluation script appends one row per episode to a ``.jsonl`` file and
then aggregates to CSV and LaTeX tables. See :doc:`/tutorials/evaluate_and_log`.

Agent configurations
--------------------

The same five building blocks run under four team compositions. In all
configurations "human" means a human operator **selects the subtask** (which
``SkillCall`` to issue) and the robot executes it via its skill executor.
The execution pipeline is identical regardless of who chose the subtask, so
Human–Human provides a fair performance upper bound.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Configuration
     - Description
   * - **Robot–Robot**
     - Both agents planned by an LLM. Primary experimental configuration.
   * - **Human–Robot**
     - One human operator + one LLM agent. Tests asymmetric teaming.
   * - **Human–Human**
     - Both operators select subtasks. No LLM involved. Upper-bound baseline.
   * - **Single Robot**
     - One arm, one planner (LLM or human). Isolates individual capability.

Task coordination classes
-------------------------

Tasks are classified by their **coordination structure**, which determines
which failure modes occur most often and which controller benefits most.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Description
   * - **Sequential-Dependent**
     - Fixed order; each step depends on the previous one (e.g. Sandwich:
       bread → bacon → cheese → tomato → bread).
   * - **Parallel-Independent**
     - Steps can happen in any order and simultaneously (e.g. Pack Grocery,
       Sort).
   * - **Gated**
     - One agent must complete a prerequisite before the other can proceed
       (e.g. Cabinet: open door first).
   * - **Tightly-Coupled**
     - Both agents must act simultaneously on the same object (e.g. Rope).
   * - **Continuous-Cooperative**
     - No discrete steps; agents maintain shared state over time (e.g. Sweep).
