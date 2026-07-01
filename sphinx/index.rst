CRIE-Bench Documentation
========================

**CRIE-Bench** (Collaborative Robot Intelligence Engine Benchmark) is a modular
framework for evaluating LLM-driven multi-agent collaboration in robot manipulation.
It provides standardised symbolic subtask interfaces, four agent configurations
(Robot–Robot, Human–Robot, Human–Human, Single Robot), pluggable planners, skills,
tasks, and structured failure and recovery logging.

.. note::

   If you are new here, start with :doc:`guides/quickstart`.
   The :doc:`guides/concepts` page explains the key abstractions before you write
   any code.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guides/installation
   guides/quickstart
   guides/concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/run_sandwich
   tutorials/add_task
   tutorials/add_skill
   tutorials/add_planner
   tutorials/agent_configurations
   tutorials/evaluate_and_log

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/controllers
   reference/planners
   reference/executors
   reference/skills
   reference/tasks
   reference/types
   reference/metrics
   reference/api

.. toctree::
   :maxdepth: 1
   :caption: About

   guides/experimental_design
   guides/contributing
