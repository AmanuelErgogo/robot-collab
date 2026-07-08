# Prompt: Draft Experimental Setup and Results for CRIE-BT

You are helping draft the experimental section of a robotics paper. Write in
concise, precise academic English. Use passive voice for methods and active
voice for findings. Avoid filler and keep each paragraph short.

## Paper Context

CRIE-BT is an LLM-driven multi-robot execution framework that combines
RoCoBench task planning with a behavior-tree runtime controller. The controller
monitors progress, records uncertainty and failures, performs bounded local
retries, triggers replanning, and emits human-facing explanations.

## Primary Methods

Only these two methods are primary paper methods:

| Method | Runner flags |
| --- | --- |
| CRIE-BT-Dialog | `--mode bt_mediated --planner-mode dialog --adapter legacy` |
| VLM/SARM-Monitor-Planner-Dialog | `--mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy` |

The VLM/SARM simulator baseline uses the same planner and executor path as
CRIE-BT. It exposes the same monitor interface intended for the real VLM/SARM
backend, but the simulator backend maps `env.get_reward_done()` to `DONE` and
executor failure feedback to `FAILED`.

In simulator results, this VLM/SARM monitor-planner baseline replaces the old
direct-feedback baseline. It is behaviorally equivalent to direct-feedback
replanning under simulator done/failure signals, but it must be named
VLM/SARM-Monitor-Planner because it logs monitor decisions and preserves the
real VLM/SARM backend interface.

## Ablations

Only include these centralized ablations when requested by a table:

| Ablation | Runner flags |
| --- | --- |
| CRIE-BT-Cent | `--mode bt_mediated --planner-mode chat --adapter legacy` |
| VLM/SARM-Monitor-Planner-Cent | `--mode vlm_sarm_monitor_planner --planner-mode chat --adapter legacy` |

Do not present direct-feedback or open-loop variants as primary methods. If
direct-feedback is mentioned, describe it only as the historical implementation
name that is now reported as the VLM/SARM simulator baseline.

## Evaluation Scope

The simulator evaluation covers four Robot-Robot tasks:

- Sandwich
- Pack Grocery
- Cabinet
- Sort

The real-world evaluation scope is collaborative medication dispensing and
collaborative cooking. Cabinet and Sort may involve more than two robot agents,
so use "robot-only multi-robot" rather than "two-robot" when describing the
full simulator task set.

## Metrics

Report these implemented metrics from `episodes.jsonl` and `summary.csv`:

- Task success rate: fraction of episodes with `sim_success=true`.
- Controller success rate: fraction of episodes with `success=true`.
- Completion time: mean `wall_time_s`.
- Token consumption: mean prompt, completion, and total tokens from
  `llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens`.
- LLM latency: mean over `llm_call_latencies_s`.
- Recovery behavior: `replans`, `local_retries`, `failure_counts`, and event
  summaries.
- Monitor decisions for the VLM/SARM baseline: `VLM_SARM_MONITOR` events and
  `payload.monitor_decision`.

Only report reactivity and hallucination rate when the corresponding annotation
fields are present:

- Reactivity: `reactivity_s`, manually or externally annotated.
- Hallucination rate: `hallucination_count / hallucination_annotation_count`.

Use placeholders with `\TODO{}` for missing numbers or annotation-only metrics.

## Result Paths

Use the canonical result root for new paper runs:

```text
results/robot_robot_sim_v1/{task_id}/{method}/episodes.jsonl
results/robot_robot_sim_v1/{task_id}/{method}/analysis/
results/robot_robot_sim_v1/{task_id}/{method}/prompts/
```

Use method directories:

```text
crie_bt_dialog
vlm_sarm_dialog
crie_bt_cent
vlm_sarm_cent
```

Treat older paths such as `results/sandwich_bt_mediated/` and
`results/c3_to_c6_runs/` as legacy artifacts.

## What To Generate

Generate the following LaTeX sections:

1. `\section{Experimental Setup}`
   - Describe CRIE-BT-Dialog and VLM/SARM-Monitor-Planner-Dialog as the two
     primary methods.
   - Describe CRIE-BT-Cent and VLM/SARM-Monitor-Planner-Cent only as
     centralized ablations.
   - Describe the four simulator tasks and the two real-world tasks.
   - Define implemented metrics and mark annotation-only metrics clearly.
   - Briefly describe simulator, robots, LLM, seeds, and episode count.

2. `\section{Results}`
   - Compare the two primary methods across simulated tasks.
   - Include centralized ablations only in a separate ablation table.
   - Use placeholders for metrics not yet collected.
   - Do not invent results for real-world VLM/SARM or learned-policy execution.

## Formatting Requirements

- Use booktabs tables with `\toprule`, `\midrule`, and `\bottomrule`.
- Use `\begin{table}[t]`, `\centering`, `\caption{}`, and `\label{tab:...}`.
- Define the following at the top of the LaTeX file:

```latex
\usepackage{booktabs}
\usepackage{xcolor}
\newcommand{\TODO}[1]{\textcolor{red}{[\textbf{TODO:} #1]}}
\newcommand{\ours}{\dag}
```

- Mark proposed-method rows with a `$^\ours$` superscript and add the footnote
  `$^\dag$ Proposed method.`
- Keep each paragraph to 3-5 sentences and let the tables carry the numbers.
