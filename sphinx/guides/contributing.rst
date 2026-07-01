Contributing
============

Pull requests are welcome. This page describes the conventions used
throughout the codebase.

Code style
----------

* Python 3.9-compatible syntax throughout ``rocobench/``.
* All public data types are frozen dataclasses with ``to_dict()`` /
  ``from_dict()`` for JSON serialisation.
* Enum values are strings (``"open_loop"``, ``"MISSED_GRASP"``) so logs are
  human-readable without import.
* No ``print()`` in library code — use the ``events`` list in the episode log
  for structured output.

Adding a task
-------------

See :doc:`/tutorials/add_task`.

Adding a skill
--------------

See :doc:`/tutorials/add_skill`.

Adding a planner
----------------

See :doc:`/tutorials/add_planner`.

Running tests
-------------

Tests do **not** require MuJoCo or an LLM API key. All simulator interactions
are replaced with ``ScriptedSkillExecutor`` and all LLM calls with fake
prompters.

.. code-block:: bash

   # Unit tests (no MuJoCo)
   python -m pytest tests/crie_bt/ -v

   # Skill tests
   python -m pytest tests/skills/ -v

   # All tests
   python -m pytest tests/ -v

.. note::

   The ``hydra`` package registers a ``pytest11`` plugin that imports
   ``antlr4``, which fails on Python 3.12+. If pytest crashes at startup,
   run tests directly:

   .. code-block:: bash

      python -c "
      from tests.crie_bt.test_sandwich_modes import *
      test_plan_mode_single_episode()
      print('OK')
      "

Building the docs
------------------

From the ``sphinx/`` directory:

.. code-block:: bash

   cd sphinx
   make html
   open _build/html/index.html
