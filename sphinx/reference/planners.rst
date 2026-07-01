Planners
========

A planner converts a task goal and environment observation into a
:class:`~rocobench.crie_bt.CollaborativePlan`.

Base interface
--------------

.. autoclass:: rocobench.crie_bt.planner.BasePlanner
   :members:
   :undoc-members:

Built-in planners
-----------------

.. autoclass:: rocobench.crie_bt.ScriptedPlanner
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rocobench.crie_bt.LLMPlannerAdapter
   :members:
   :undoc-members:
   :show-inheritance:

RoCoBench LLM bridge
---------------------

.. autoclass:: rocobench.crie_bt.LegacyPromptPlanner
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rocobench.crie_bt.LegacyActionPlanner
   :members:
   :undoc-members:
   :show-inheritance:

Task-specific adapters
-----------------------

.. autoclass:: rocobench.crie_bt.PackGroceryCRIEPlanner
   :members:
   :undoc-members:
   :show-inheritance:

Communication modes (``planner_mode``)
---------------------------------------

``LegacyPromptPlanner`` accepts a ``planner_mode`` argument that controls how
the LLM is queried:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - Behaviour
   * - ``"plan"``
     - Single centralised prompt, no history. One LLM call per
       ``generate_plan`` invocation. Fastest; use as a no-history baseline.
   * - ``"chat"``
     - Single centralised prompt with the full conversation history appended.
       Grows with each replan. Primary centralised condition.
   * - ``"dialog"``
     - Per-agent turn-taking via ``DialogPrompter``. Each agent sees the
       other agent's last message before committing. Richer coordination but
       prone to single-agent parse failures.
