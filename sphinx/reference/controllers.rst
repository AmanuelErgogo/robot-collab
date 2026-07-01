Controllers
===========

Controllers own the episode loop: they call the planner, dispatch subtasks
to the executor, read :class:`~rocobench.crie_bt.ExecutionFeedback`, and
decide what to do next.

All three controllers share the same constructor signature:

.. code-block:: python

   controller = SomeController(
       planner=planner,          # BasePlanner
       executor=executor,        # BaseSkillExecutor
       uncertainty_mode="heuristic",   # "none" | "heuristic" | "ensemble_variance"
       max_retries=1,
   )

   log = controller.run_episode(env, task_goal="...", max_steps=50)

.. autoclass:: rocobench.crie_bt.OpenLoopController
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rocobench.crie_bt.DirectFeedbackController
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rocobench.crie_bt.BTMediatedController
   :members:
   :undoc-members:
   :show-inheritance:

Episode log keys
-----------------

Every ``run_episode`` call returns a ``dict[str, Any]`` with the following
guaranteed keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``mode``
     - :class:`~rocobench.crie_bt.ExecutionMode` value (``"open_loop"``, …)
   * - ``task_goal``
     - The ``task_goal`` string passed to ``run_episode``.
   * - ``success``
     - ``True`` if all plan steps completed without an active failure.
   * - ``steps``
     - Total executor ``.step()`` ticks across the episode.
   * - ``planner_calls``
     - Number of ``planner.generate_plan()`` calls.
   * - ``replans``
     - Number of times the controller triggered a replan after failure.
   * - ``local_retries``
     - Executor-level retries below the LLM layer.
   * - ``failure_counts``
     - ``dict[FailureCode.value, int]`` — failure code histogram.
   * - ``completed_subtasks``
     - Plan steps that returned ``BTStatus.SUCCESS``.
   * - ``failed_subtasks``
     - Plan steps that returned ``BTStatus.FAILURE``.
   * - ``subtask_results``
     - ``list[dict]`` — one ``ExecutionFeedback.to_dict()`` per step.
   * - ``explanations``
     - ``list[str]`` — human-readable notes added by the controller.
   * - ``events``
     - ``list[RuntimeEvent.to_dict()]`` — structured event log.

Choosing a controller
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Controller
     - When to use
   * - ``OpenLoopController``
     - Baseline. Fastest (one LLM call per episode). Use when you want to
       evaluate raw LLM planning quality without replanning.
   * - ``DirectFeedbackController``
     - When you want the LLM to recover from failures but do not need
       structured BT mediation. Good middle ground.
   * - ``BTMediatedController``
     - The proposed method. Use when you want the BT to route failures to the
       right recovery strategy (local retry, replan, or human escalation).

Build helper
-------------

.. autofunction:: rocobench.crie_bt.controllers.build_controller
