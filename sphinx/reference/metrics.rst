Metrics
=======

CRIE-Bench computes two primary metrics and several secondary efficiency and
recovery metrics. All metrics are computed by ``aggregate()`` in
``scripts/eval_sandwich_open_loop.py``.

Primary metrics
---------------

Action Success Rate (ASR)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{ASR} = \frac{\text{episodes where } \texttt{success} = \text{True}}{n}

``success = True`` when every plan step returned ``BTStatus.SUCCESS`` without
an active failure. This indicates the LLM produced a parseable EXECUTE block
**and** the skill executor physically completed the action.

**When to use**: Primary metric for open-loop conditions, where Task
Completion Rate is structurally near zero (only one subtask executes per
episode, but the task needs ~10).

Task Completion Rate (TCR)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{TCR} = \frac{\text{episodes where } \texttt{sim\_success} = \text{True}}{n}

``sim_success = True`` when ``env.get_reward_done()`` returned ``done=True``
— the full task was assembled in the simulator.

**When to use**: Primary metric for ``DirectFeedbackController`` and
``BTMediatedController``, which loop until the task is done or the step
budget is exhausted.

Confidence intervals
~~~~~~~~~~~~~~~~~~~~~

Both ASR and TCR are accompanied by 95 % Wilson score confidence intervals.
Wilson intervals are tighter than the normal approximation for small *n* and
remain valid when the proportion is 0 or 1.

.. math::

   \tilde{p} = \frac{k + z^2/2}{n + z^2}, \quad
   \delta = \frac{z}{n+z^2}\sqrt{\frac{k(n-k)}{n} + \frac{z^2}{4}}

   \text{CI} = [\tilde{p} - \delta,\ \tilde{p} + \delta], \quad z = 1.96

Secondary metrics
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Definition
   * - **Steps**
     - Mean executor ``.step()`` ticks per episode ± std. Each tick = one
       simulator time step.
   * - **Wall time**
     - Mean episode wall-clock time (seconds) ± std. Includes LLM latency,
       RRT planning, and MuJoCo simulation.
   * - **LLM latency**
     - Mean per-call LLM response time (seconds). Excludes simulation and RRT.
   * - **Planner errors**
     - Count of episodes that exhausted ``num_replans`` without a valid EXECUTE
       block. Reported as ``k/n``.
   * - **Replans**
     - Mean number of LLM replan calls per episode.
   * - **Local retries**
     - Mean executor-level retries (below the LLM layer) per episode.

Recovery metrics (feedback-loop controllers only)
--------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Definition
   * - **Recovery Rate**
     - Among episodes with ≥ 1 failure, the fraction that ultimately succeeded.
       Measures how well the controller handles adversity.
   * - **Steps-to-Recovery**
     - Mean executor steps from first failure to next successful subtask.
       Lower is better.
   * - **Unnecessary Replans**
     - Replans triggered when no failure was active. Proxy for over-sensitivity
       of the failure detector or BT trigger condition.

Failure code breakdown
----------------------

Every episode logs ``failure_counts``, a histogram over
:class:`~rocobench.crie_bt.FailureCode` values:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Code
     - Trigger
   * - ``PLANNER_ERROR``
     - LLM never returned a parseable EXECUTE block within ``num_replans``.
   * - ``POSTCONDITION_FAILED``
     - RRT ran but the skill's postcondition was not satisfied.
   * - ``MISSED_GRASP``
     - Gripper closed but object not detected as held.
   * - ``NO_PROGRESS``
     - Executor exceeded step budget without advancing.
   * - ``WRONG_OBJECT``
     - Robot picked the wrong object.
   * - ``WRONG_TARGET``
     - Robot placed at the wrong target.
   * - ``SAFETY_CONFLICT``
     - Planned action would cause a collision.
   * - ``TIMEOUT``
     - Episode exceeded wall-clock budget.
   * - ``UNKNOWN``
     - Catch-all for unclassified executor exceptions.

Metric selection guide
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Controller
     - Primary metric
   * - ``OpenLoopController``
     - **ASR** — TCR is structurally ~0% for multi-step tasks.
   * - ``DirectFeedbackController``
     - **TCR** and **Recovery Rate**.
   * - ``BTMediatedController``
     - **TCR**, **Recovery Rate**, **Unnecessary Replans** (BT sensitivity).
   * - Human–Human baseline
     - **TCR** — the performance ceiling for each task.
