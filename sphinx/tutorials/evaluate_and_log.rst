Tutorial: Evaluation and Logging
=================================

Every call to ``controller.run_episode(...)`` returns a plain dict. The
evaluation pipeline collects these dicts, aggregates them, and writes
paper-ready outputs.

Episode log structure
----------------------

.. code-block:: python

   {
       "mode": "open_loop",          # ExecutionMode value
       "task_goal": "assemble sandwich",
       "success": True,              # all planned steps succeeded
       "sim_success": False,         # env.get_reward_done() returned done
       "steps": 1,                   # total executor ticks
       "planner_calls": 1,
       "replans": 0,
       "local_retries": 0,
       "failure_counts": {
           "MISSED_GRASP": 1,        # FailureCode → count
       },
       "completed_subtasks": 1,
       "failed_subtasks": 0,
       "subtask_results": [...],     # per-step ExecutionFeedback dicts
       "explanations": [],
       "wall_time_s": 9.3,
       "llm_calls": 1,
       "llm_avg_latency_s": 4.9,
   }

``success`` vs ``sim_success``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``success`` — the CRIE-BT internal signal: every plan step returned
  ``BTStatus.SUCCESS`` without an active failure. Indicates the LLM produced
  a valid plan and the skill executor physically completed it.
* ``sim_success`` — ``env.get_reward_done()`` returned ``done=True``. Indicates
  the full task was assembled. For open-loop mode this is structurally near
  zero because only one subtask executes per episode.

Use ``success`` (ASR) as the primary metric for open-loop conditions; use
``sim_success`` (TCR) for feedback-loop conditions.

Running a multi-episode evaluation
------------------------------------

Use the evaluation script directly:

.. code-block:: bash

   python scripts/eval_sandwich_open_loop.py \
       --modes plan chat dialog \
       --num-episodes 5 \
       --seeds 0 1 2 \
       --output-dir results/my_run

Or call the Python API:

.. code-block:: python

   from scripts.eval_sandwich_open_loop import run_one_episode, aggregate, write_outputs

   rows = []
   for seed in [0, 1, 2]:
       for ep in range(5):
           row = run_one_episode(mode="plan", comm="chat", seed=seed, episode=ep)
           rows.append(row)

   stats = aggregate(rows, modes=["plan"])
   write_outputs(rows, stats, output_dir="results/my_run")

Aggregated metrics
-------------------

``aggregate()`` computes per-mode statistics:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Definition
   * - ``action_success_rate``
     - ``success`` count / n
   * - ``asr_ci_lo``, ``asr_ci_hi``
     - Wilson score 95 % confidence interval
   * - ``task_completion_rate``
     - ``sim_success`` count / n
   * - ``avg_steps_all``
     - Mean executor steps across all episodes
   * - ``avg_wall_time_s``
     - Mean episode wall-clock time (seconds)
   * - ``avg_llm_latency_s``
     - Mean per-call LLM response time (excludes sim)
   * - ``planner_errors``
     - Episodes that exhausted ``num_replans`` without a valid plan
   * - ``avg_replans``
     - Mean LLM replan calls per episode

Output files
-------------

.. code-block:: text

   results/my_run/
   ├── episodes.jsonl     ← one JSON line per episode
   ├── summary.csv        ← aggregated stats (spreadsheet-ready)
   ├── summary.json       ← same, as JSON
   ├── table.txt          ← ASCII table
   ├── table.tex          ← LaTeX booktabs table for Overleaf
   └── failures.json      ← per-mode failure code breakdown

LaTeX table sample
-------------------

``table.tex`` is ready to paste into Overleaf:

.. code-block:: latex

   \begin{table}[t]
     \centering
     \caption{Sandwich task: plan, chat, and dialog communication modes
              with open-loop execution (Gemini 2.5 Flash, $n=15$ per mode).}
     \label{tab:sandwich_open_loop}
     \begin{tabular}{lrrrrrr}
       \toprule
       \textbf{Mode} & \textbf{ASR (\%)} & \textbf{95\% CI} &
       \textbf{Steps} & \textbf{Time (s)} & \textbf{LLM Lat (s)} &
       \textbf{P.Err} \\
       \midrule
       Plan (Centralised) & 100.0 & [80, 100] & $1.0 \pm 0.0$ & $9.3 \pm 0.3$ & 4.9 & 0/15 \\
       Chat (w/ History)  & 100.0 & [80, 100] & $1.0 \pm 0.0$ & $15.6 \pm 1.6$ & 12.5 & 0/15 \\
       Dialog (Multi-agent) & 86.7 & [62, 96] & $0.9 \pm 0.4$ & $18.5 \pm 11.7$ & 16.0 & 2/15 \\
       \bottomrule
     \end{tabular}
   \end{table}

Failure analysis
-----------------

``failures.json`` breaks down failure codes per mode:

.. code-block:: json

   {
     "dialog": {
       "PLANNER_ERROR": 2,
       "POSTCONDITION_FAILED": 0
     }
   }

Use this to diagnose whether failures are LLM-side (``PLANNER_ERROR``) or
execution-side (``MISSED_GRASP``, ``NO_PROGRESS``, etc.).

Re-running table generation
-----------------------------

If you already have ``episodes.jsonl`` and only want to regenerate tables
(e.g., after changing the metric definition):

.. code-block:: bash

   python scripts/eval_sandwich_open_loop.py \
       --from-jsonl results/my_run/episodes.jsonl \
       --output-dir results/my_run
