# Prompt: Convert CRIE-BT Experimental Design to Paper Sections (Overleaf / LaTeX)

---

You are a scientific writing assistant helping to draft the **Experiments** and **Results** sections of a robotics paper for an IEEE/ACM venue (e.g., ICRA, CoRL, or RA-L). Write in clear, precise academic English. Use passive voice for methods, active for findings. Do not pad sentences.

## Paper context

We propose **CRIE-BT** (Collaborative Robot Intelligence Engine with Behavior Trees), a framework for LLM-driven multi-robot task execution with structured failure recovery. The key contribution is using a Behavior Tree (BT) to mediate between LLM planning and physical execution: the BT monitors uncertainty estimates from the skill executor and decides whether to replan via LLM, retry locally, or escalate to a human operator — rather than routing every failure directly back to the LLM.

We compare CRIE-BT against two baselines and evaluate across multiple tasks, team compositions, and LLM communication strategies.

---

## What to generate

Generate the following LaTeX sections, in order. Use `booktabs` for all tables, `\toprule / \midrule / \bottomrule`. Use `\TODO{}` as a macro (define it as `\newcommand{\TODO}[1]{\textcolor{red}{[TODO: #1]}}`) wherever a result, figure, or number needs to be filled in later.

### Sections to write

1. **`\section{Experimental Setup}`**
   - **`\subsection{Conditions}`** — Describe the 2×3 condition matrix:
     - Feedback modes (rows): No-Feedback (open-loop baseline), With-Feedback (direct LLM replanning), Feedback+BT (our method)
     - Communication modes (columns): Centralised-with-history (single LLM prompt with conversation history), Dialog (per-agent turn-taking)
     - Label conditions C1–C6. Emphasise that C5 and C6 are the proposed method.
     - Include a small `tabular` showing the 3×2 matrix with condition labels.

   - **`\subsection{Agent Configurations}`** — Describe the four team compositions:
     - Robot–Robot: both arms LLM-planned (primary)
     - Human–Robot: one human operator selects subtasks, one arm LLM-planned (asymmetric teaming)
     - Human–Human: both operators select subtasks, no LLM (upper-bound reference baseline)
     - Single-Robot: one arm, LLM-planned (isolates individual capability)
     - Clarify that "human" means the operator selects a subtask (e.g., PICK bread\_slice1) and the robot executes autonomously via RRT motion planning — the execution pipeline is identical across all configurations, ensuring a fair comparison.

   - **`\subsection{Tasks}`** — Describe all six tasks and their coordination class. Include a table:

     | Task | Agents | Coordination Class | Action ordering |
     |---|---|---|---|
     | Sandwich | 2 | Sequential-Dependent | Strict recipe order |
     | Pack Grocery | 2 | Parallel-Independent | Flexible |
     | Cabinet | 3 | Gated | Prerequisite-dependent |
     | Sort | 3 | Parallel-Independent | Flexible |
     | Sweep | 2 | Continuous-Cooperative | None (continuous) |
     | Rope | 2 | Tightly-Coupled | Simultaneous bimanual |

     Define each coordination class in one sentence. Note that Cabinet and Sort have three agents, which dialog mode handles with an additional per-agent turn.

   - **`\subsection{Metrics}`** — Define:
     - **ASR** (Action Success Rate): fraction of episodes where the LLM produced a parseable action plan and the RRT skill executor physically completed it. Used as primary metric for No-Feedback conditions (where Task Completion Rate is structurally near zero because only one LLM-planned action executes per episode).
     - **TCR** (Task Completion Rate): fraction of episodes where the full task was completed (simulator `done=True`). Primary metric for With-Feedback and Feedback+BT conditions.
     - **95% CI** for both, computed via Wilson score interval.
     - Efficiency: mean steps, wall-clock time, LLM latency per call.
     - Recovery (With-Feedback and Feedback+BT only): Recovery Rate (fraction of episodes with ≥1 failure that still succeeded), Steps-to-Recovery, Unnecessary Replans.
     - Include one sentence justifying why ASR and TCR are complementary rather than redundant.

   - **`\subsection{Implementation Details}`** — One short paragraph:
     - Simulator: MuJoCo. Robots: UR5e-Robotiq (Chad) and Panda (Dave).
     - LLM: Gemini 2.5 Flash via Vertex AI.
     - Motion planning: RRT via the RoCoBench skill executor.
     - Seeds: {0, 1, 2}. Episodes per seed: \TODO{N}. Total per condition-task pair: \TODO{N\_total}.
     - `num\_replans=2` for all LLM conditions.

2. **`\section{Results}`**

   - **`\subsection{No-Feedback Baseline (C1 vs C2)}`**
     - One paragraph of framing text with \TODO{} for all numbers.
     - One `booktabs` table: rows = {C1 Centralised, C2 Dialog}, columns = {ASR (\%), ASR 95\% CI, Steps, Time (s), LLM Lat (s), P.Err}. Fill every cell with `\TODO{}`.
     - Placeholder sentence: "Centralised achieves \TODO{}\% ASR versus \TODO{}\% for Dialog ($p = \TODO{}$), with Dialog incurring \TODO{}$\times$ higher LLM latency due to per-agent turn overhead."

   - **`\subsection{Effect of Feedback (C1/C2 vs C3/C4)}`**
     - One paragraph framing. Table: rows = {C1, C2, C3, C4}, columns = {ASR (\%), TCR (\%), Replans, Recovery Rate (\%), Time (s)}. All cells `\TODO{}`.
     - Placeholder: "Adding direct feedback improves TCR from \TODO{}\% to \TODO{}\%, confirming that single-step open-loop execution is insufficient for multi-step tasks."

   - **`\subsection{CRIE-BT vs Direct Feedback (C3/C4 vs C5/C6)}`**
     - One paragraph framing. Table: rows = {C3, C4, C5 (ours), C6 (ours)}, columns = {TCR (\%), Recovery Rate (\%), Unnecessary Replans, Steps-to-Recovery, Time (s)}. All cells `\TODO{}`.
     - Placeholder: "Feedback+BT reduces unnecessary replans by \TODO{}\% and improves recovery rate by \TODO{} percentage points, at a wall-clock overhead of \TODO{} s per episode."

   - **`\subsection{Team Composition (Robot–Robot vs Human–Robot vs Human–Human)}`**
     - One paragraph framing. Table: rows = {Robot–Robot (C5), Human–Robot (C5), Human–Human (reference)}, columns = {Task, TCR (\%), ASR (\%), Steps}. All cells `\TODO{}`. Note that Human–Human uses no LLM.
     - Placeholder: "Human–Human achieves \TODO{}\% TCR, setting the performance ceiling. CRIE-BT (C5, Robot–Robot) reaches \TODO{}\% of this ceiling."

   - **`\subsection{Task Analysis by Coordination Class}`**
     - One paragraph framing. Table: rows = 6 tasks, columns = {Class, C1 TCR, C5 TCR, Δ TCR, Recovery Rate (C5)}. All cells `\TODO{}`.
     - Placeholder: "Sequential-Dependent tasks benefit most from Feedback+BT (\TODO{} pp improvement), while Parallel-Independent tasks show smaller gains (\TODO{} pp), consistent with the lower replanning need when ordering is flexible."

   - **`\subsection{Ablation: Communication Mode}`**
     - One paragraph comparing Centralised vs Dialog within each feedback level. Table: rows = {C1 vs C2, C3 vs C4, C5 vs C6}, columns = {TCR Cent, TCR Dialog, LLM Lat Cent, LLM Lat Dialog, P.Err Cent, P.Err Dialog}. All cells `\TODO{}`.
     - Placeholder: "Dialog yields \TODO{} pp higher TCR on Tightly-Coupled tasks (Rope) where per-agent negotiation aligns physical roles, but underperforms Centralised on Sequential-Dependent tasks due to parse failures (\TODO{}/\TODO{} episodes)."

---

## Formatting requirements

- Use `\begin{table}[t]` with `\centering`, `\caption{}`, `\label{tab:...}`.
- Every table caption must state: task(s), LLM, number of episodes, seeds.
- Every `\TODO{}` must include a short hint, e.g. `\TODO{ASR for C1, sandwich}`.
- Define at the top of the LaTeX file:
  ```latex
  \usepackage{booktabs}
  \usepackage{xcolor}
  \newcommand{\TODO}[1]{\textcolor{red}{[\textbf{TODO:} #1]}}
  \newcommand{\ours}{\dag}  % dagger marker for our method rows
  ```
- Mark all C5/C6 rows in tables with a $^\ours$ superscript and add a footnote: "$^\dag$ Proposed method."
- Use `\pm` for standard deviations. Use `[lo, hi]` notation for Wilson CIs.
- Section and subsection labels: `\label{sec:setup}`, `\label{sec:results}`, `\label{subsec:nofeedback}`, etc.

---

## Tone and style notes

- Do not write "In this section we..." or "As can be seen from...".
- State results directly: "C5 achieves..." not "It can be observed that C5 achieves...".
- Keep each paragraph to 3–5 sentences. Let tables carry the numbers; prose carries interpretation.
- "Centralised" and "Dialog" are proper nouns in this paper; capitalise them.
- Spell out "Behavior Tree" on first use, then use "BT".
- Do not use "leverage", "showcase", "delve", or "robust" as filler.
