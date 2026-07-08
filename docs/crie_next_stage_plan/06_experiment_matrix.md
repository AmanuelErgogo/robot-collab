# Experiment Matrix

## Step 1: robot-robot simulation

Purpose: test whether CRIE-BT with an explicit coded simulator monitor improves robustness over monolithic VLM self-monitoring when both use the same RRT skill backend.

| Condition | Team | Coordination | Skill | Monitor | Planner input | Environment |
|---|---|---|---|---|---|---|
| `VLM-RR-Cent` | robot-robot | centralized | RRT | VLM self-monitor | image/status/history | sim |
| `VLM-RR-Dialog` | robot-robot | dialog/distributed | RRT | VLM self-monitor | image/status/history/dialogue | sim |
| `CRIE-BT-RR-Cent` | robot-robot | centralized | RRT | coded simulator monitor | image/status/history + monitor signal | sim |
| `CRIE-BT-RR-Dialog` | robot-robot | dialog/distributed | RRT | coded simulator monitor | image/status/history/dialogue + monitor signal | sim |

Recommended tasks:

- `sandwich`
- `pack`
- `cabinet`
- `sort`

Recommended protocol:

- 10 episodes × 3 seeds per condition per task as a first pass.
- Increase to 20–30 episodes per condition per task after debugging.

## Step 2: human-robot simulation

Purpose: test whether the same architecture works when one collaborator is an actual human controlling a simulated agent through terminal commands.

| Condition | Team | Coordination | Skill | Monitor | Human interface | Environment |
|---|---|---|---|---|---|---|
| `VLM-HR-Cent` | human-robot | centralized | RRT robot + terminal human | VLM self-monitor | terminal | sim |
| `VLM-HR-Dialog` | human-robot | dialog/distributed | RRT robot + terminal human | VLM self-monitor | terminal | sim |
| `CRIE-BT-HR-Cent` | human-robot | centralized | RRT robot + terminal human | coded simulator monitor | terminal | sim |
| `CRIE-BT-HR-Dialog` | human-robot | dialog/distributed | RRT robot + terminal human | coded simulator monitor | terminal | sim |

Recommended tasks:

- medication dispensing in simulator;
- cooking in simulator.

Recommended protocol:

- Start with 1–3 pilot users to debug the interface.
- Then run repeated trials with one expert user for system validation.
- Later run a human-subject pilot if needed.

## Step 3: real-world human-robot evaluation

Purpose: test the final system with learned visual skills and SARM progress monitoring.

| Condition | Team | Coordination | Skill | Monitor | Human interface | Environment |
|---|---|---|---|---|---|---|
| `VLM-HR-Cent` | human-robot | centralized | learned visual skill | VLM self-monitor | terminal/speech | real |
| `VLM-HR-Dialog` | human-robot | dialog/distributed | learned visual skill | VLM self-monitor | terminal/speech | real |
| `CRIE-BT-HR-Cent` | human-robot | centralized | learned visual skill | SARM progress monitor | terminal/speech | real |
| `CRIE-BT-HR-Dialog` | human-robot | dialog/distributed | learned visual skill | SARM progress monitor | terminal/speech | real |

Recommended tasks:

- collaborative medication dispensing;
- collaborative cooking.

## Minimal ablations

| Ablation | Purpose |
|---|---|
| `CRIE-BT without monitor` | test whether BT alone helps without explicit progress signal. |
| `CRIE-BT with coded monitor but no local retry` | isolate local retry from monitoring. |
| `CRIE-BT with oracle text planner input` | upper-bound performance; must be labeled oracle. |
| `Baseline with privileged state` | optional upper bound; not a fair baseline. |

## Primary hypotheses

H1: In robot-robot simulation, CRIE-BT with coded simulator monitoring will achieve higher task success and fewer unrecovered failures than VLM self-monitoring baselines.

H2: In human-robot simulation, CRIE-BT will reduce unnecessary replanning and improve recovery under human delay, rejection, or counter-proposal.

H3: In real-world HRC, replacing the coded monitor with SARM will preserve the architectural benefits of explicit progress monitoring while avoiding privileged simulator signals.
