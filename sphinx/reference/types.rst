Data Types
==========

All CRIE-BT data types are frozen dataclasses. Every type implements
``to_dict()`` / ``from_dict()`` for JSON serialisation.

Core plan types
---------------

.. autoclass:: rocobench.crie_bt.SkillCall
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.PlanStep
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.CollaborativePlan
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.SubtaskCommand
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.RoleAssignment
   :members:
   :undoc-members:

Feedback types
--------------

.. autoclass:: rocobench.crie_bt.ExecutionFeedback
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.ProgressState
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.UncertaintyState
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.FailureState
   :members:
   :undoc-members:

Context and decision types
---------------------------

.. autoclass:: rocobench.crie_bt.ExecutionContext
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.BTDecision
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.RuntimeEvent
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.ControllerResult
   :members:
   :undoc-members:

Enumerations
------------

.. autoclass:: rocobench.crie_bt.BTStatus
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.ExecutionMode
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.FailureCode
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.ProgressStage
   :members:
   :undoc-members:

.. autoclass:: rocobench.crie_bt.RuntimeDecision
   :members:
   :undoc-members:
