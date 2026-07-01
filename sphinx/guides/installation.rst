Installation
============

Requirements
------------

* Python 3.9 – 3.11 (3.13 not supported; ``antlr4`` in the RoCo runtime
  requires ``typing.io`` which was removed in 3.12).
* `Conda <https://docs.conda.io>`_ recommended — MuJoCo and the RRT planner
  have C extensions that are easier to manage in an isolated environment.
* A GPU is not required for the simulator but speeds up learned-skill
  inference significantly.

Step 1 — Create the conda environment
--------------------------------------

.. code-block:: bash

   conda create -n roco python=3.10 -y
   conda activate roco

Step 2 — Install the package
-----------------------------

From the repository root:

.. code-block:: bash

   pip install -e ".[visualization]"

The ``visualization`` extra adds ``open3d`` for point-cloud rendering. It is
optional; all evaluation scripts run without it.

Step 3 — MuJoCo
----------------

MuJoCo is bundled with ``mujoco`` (>=2.3). If you see rendering errors, set
the off-screen backend before running:

.. code-block:: bash

   export MUJOCO_GL=egl        # headless server
   # or
   export MUJOCO_GL=osmesa     # software rendering

Step 4 — LLM credentials (optional)
-------------------------------------

CRIE-Bench can run fully without an LLM using :class:`~rocobench.crie_bt.ScriptedPlanner`.
For real LLM evaluation, set one of the following:

**Google Gemini via Vertex AI**

.. code-block:: bash

   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
   # If your conda env lacks google.genai, point to a Python that has it:
   export GOOGLE_GENAI_PYTHON_BIN=/path/to/python3.13

**OpenAI**

.. code-block:: bash

   export OPENAI_API_KEY=sk-...

Step 5 — Verify
---------------

.. code-block:: bash

   python -c "from rocobench.crie_bt import OpenLoopController, ScriptedPlanner, ScriptedSkillExecutor; print('OK')"
