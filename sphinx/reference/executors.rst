Skill Executors
===============

Executors receive a :class:`~rocobench.crie_bt.SkillCall` and drive the robot
through the physical action, returning
:class:`~rocobench.crie_bt.ExecutionFeedback` on every simulator tick.

Base interface
--------------

.. autoclass:: rocobench.crie_bt.executor.BaseSkillExecutor
   :members:
   :undoc-members:

Built-in executors
-------------------

.. autoclass:: rocobench.crie_bt.ScriptedSkillExecutor
   :members:
   :undoc-members:
   :show-inheritance:

   **Supported scenarios**

   Pass one of these strings as ``scenario``:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - ``scenario``
        - Simulated failure
      * - ``"none"``
        - Always succeeds after ``success_after_steps`` ticks.
      * - ``"missed_grasp"``
        - Returns ``MISSED_GRASP`` failure for the first ``fail_attempts`` attempts.
      * - ``"slippage"``
        - Returns ``SLIPPAGE`` failure.
      * - ``"no_progress"``
        - Returns ``NO_PROGRESS`` failure.
      * - ``"target_occupied"``
        - Returns ``TARGET_OCCUPIED`` failure.
      * - ``"human_interrupt"``
        - Succeeds immediately (simulates human taking over).
      * - ``"low_confidence"``
        - Returns ``RUNNING`` with low confidence on the first tick, then succeeds.

.. autoclass:: rocobench.crie_bt.RRTSkillExecutor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rocobench.crie_bt.LearnedSkillExecutor
   :members:
   :undoc-members:
   :show-inheritance:

RoCoBench adapter
------------------

.. autoclass:: rocobench.crie_bt.LegacyTaskRRTExecutorAdapter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rocobench.crie_bt.PackGroceryRRTExecutorAdapter
   :members:
   :undoc-members:
   :show-inheritance:

ExecutionFeedback structure
----------------------------

Every ``executor.step()`` call returns an
:class:`~rocobench.crie_bt.ExecutionFeedback` with three orthogonal signals:

.. code-block:: python

   feedback.status        # BTStatus.RUNNING | SUCCESS | FAILURE
   feedback.progress      # ProgressState — which stage the skill is at
   feedback.uncertainty   # UncertaintyState — confidence estimate
   feedback.failure       # FailureState — is_failure, failure_code, message

See :doc:`/reference/types` for full field documentation.

Failure detection
------------------

.. autoclass:: rocobench.crie_bt.failure.FailureDetector
   :members:
   :undoc-members:

Uncertainty estimation
-----------------------

.. autoclass:: rocobench.crie_bt.uncertainty.UncertaintyEstimator
   :members:
   :undoc-members:
