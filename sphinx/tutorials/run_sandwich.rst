Tutorial: Running the Sandwich Task
=====================================

The sandwich task is the primary benchmark in CRIE-Bench. Two robot arms
(Chad and Dave) must assemble a bacon sandwich by stacking five items in
strict order: ``bread_slice1 → bacon → cheese → tomato → bread_slice2``.

It tests **Sequential-Dependent** coordination — the single hardest class for
open-loop planners because a wrong action order cannot be recovered without a
full reset.

Prerequisites
-------------

* ``roco`` conda environment with MuJoCo installed (see :doc:`/guides/installation`).
* (Optional) LLM credentials for real planning.

Step 1 — Dry run (no LLM, no MuJoCo)
--------------------------------------

Run with ``--dry-run`` to confirm the pipeline works before touching the
simulator:

.. code-block:: bash

   python scripts/eval_sandwich_open_loop.py --dry-run

Expected output::

   [dry-run] plan   ep=0 s=0 success=True  steps=1 t=0.01s
   [dry-run] chat   ep=0 s=0 success=True  steps=1 t=0.01s
   [dry-run] dialog ep=0 s=0 success=True  steps=1 t=0.01s
   Wrote results to results/sandwich_open_loop/

Step 2 — Tutorial script (fake components, verbose output)
-----------------------------------------------------------

The tutorial script runs all three communication modes with fake LLM
components and prints a side-by-side comparison:

.. code-block:: bash

   python scripts/tutorial_sandwich_modes.py

This shows the exact prompt/response cycle for each mode without an API call.
Useful for understanding how Centralised-with-history and Dialog differ before
running the real evaluation.

Step 3 — Real evaluation (MuJoCo + Gemini)
-------------------------------------------

.. code-block:: bash

   export MUJOCO_GL=egl
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
   export GOOGLE_GENAI_PYTHON_BIN=/path/to/python3.13  # if roco env lacks google.genai

   conda run -n roco python scripts/eval_sandwich_open_loop.py \
       --modes plan chat dialog \
       --num-episodes 5 \
       --seeds 0 1 2 \
       --model gemini-2.5-flash \
       --output-dir results/sandwich_open_loop

Step 4 — Inspect results
-------------------------

The script writes five files:

.. code-block:: text

   results/sandwich_open_loop/
   ├── episodes.jsonl    ← one row per episode (raw data)
   ├── summary.csv       ← aggregated per-mode stats
   ├── summary.json      ← same, as JSON
   ├── table.txt         ← ASCII table for terminal inspection
   └── table.tex         ← LaTeX booktabs table ready for Overleaf

Print the ASCII table:

.. code-block:: bash

   cat results/sandwich_open_loop/table.txt

Expected output (n=15/mode, Gemini 2.5 Flash)::

   +----------------------+----------+--------------+------------+-------------+
   | Mode                 |  ASR (%) |   ASR 95% CI |      Steps |    Time (s) |
   +----------------------+----------+--------------+------------+-------------+
   | Plan (Centralised)   |    100.0 | [79.6,100.0] |    1.0±0.0 |     9.3±0.3 |
   | Chat (w/ History)    |    100.0 | [79.6,100.0] |    1.0±0.0 |    15.6±1.6 |
   | Dialog (Multi-agent) |     86.7 |  [62.1,96.3] |    0.9±0.4 |   18.5±11.7 |
   +----------------------+----------+--------------+------------+-------------+

Understanding the metrics
--------------------------

* **ASR** (Action Success Rate) is the primary metric for the open-loop
  controller. It measures whether the LLM produced a valid action plan
  *and* the RRT executor physically completed it.
* **TCR** (Task Completion Rate) would be ~0% for all modes under open-loop
  because assembling a sandwich requires ~10 sequential actions; the
  open-loop controller executes only one per episode.
* Use ``DirectFeedbackController`` or ``BTMediatedController`` to obtain
  meaningful TCR results. See :doc:`/tutorials/evaluate_and_log`.

Re-running from saved episodes
-------------------------------

If you already have ``episodes.jsonl`` from a previous run, regenerate
the tables without re-simulating:

.. code-block:: bash

   python scripts/eval_sandwich_open_loop.py \
       --from-jsonl results/sandwich_open_loop/episodes.jsonl \
       --output-dir results/sandwich_open_loop
