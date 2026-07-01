Experimental Design
====================

See :doc:`/reference/metrics` for metric definitions and
:doc:`/tutorials/evaluate_and_log` for the evaluation pipeline.

The full experimental design is documented in
``docs/crie_bt_experimental_design.md`` in the repository root.

Condition matrix
-----------------

Six primary conditions from crossing feedback mode × communication mode:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Centralised (w/ history)
     - Dialog
   * - **No-Feedback** (baseline)
     - C1
     - C2
   * - **With-Feedback**
     - C3
     - C4
   * - **Feedback+BT** *(ours)*
     - C5
     - C6

Agent configurations
---------------------

All six conditions run under four team compositions:

* **Robot–Robot** — both arms LLM-planned (primary).
* **Human–Robot** — one human operator, one LLM agent.
* **Human–Human** — both operators; no LLM; upper-bound reference.
* **Single Robot** — one arm only; LLM or human.

Tasks and coordination classes
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Task
     - Class
     - Key property
   * - Sandwich
     - Sequential-Dependent
     - Strict recipe order; wrong step = unrecoverable without reset.
   * - Pack Grocery
     - Parallel-Independent
     - Any item, any order; collision avoidance is the main challenge.
   * - Cabinet
     - Gated
     - Door must open before placement; deadlock risk.
   * - Sort
     - Parallel-Independent
     - Three agents; category-based target assignment.
   * - Sweep
     - Continuous-Cooperative
     - No discrete steps; only feedback-loop controllers are meaningful.
   * - Rope
     - Tightly-Coupled
     - Simultaneous bimanual; synchronisation is critical.
