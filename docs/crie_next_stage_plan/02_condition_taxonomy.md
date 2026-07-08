# Condition Taxonomy and Naming

## Core axes

Each experiment condition should be specified by six axes.

| Axis | Values | Meaning |
|---|---|---|
| controller_family | `VLM`, `CRIE-BT` | Monolithic VLM planner baseline or proposed BT-mediated controller. |
| team_type | `RR`, `HR` | Robot-robot or human-robot team. |
| coordination_mode | `Cent`, `Dialog` | Centralized control or dialog/distributed coordination. |
| skill_backend | `RRT`, `LearnedSkill` | Low-level execution backend. |
| monitor_backend | `VLM-self`, `CodedSim`, `SARM` | Progress monitoring source. |
| environment | `sim`, `real` | Simulation or real-world setup. |

## Paper-facing condition matrix

| Family | Team | Centralized | Dialog / Distributed |
|---|---|---|---|
| Baseline | Robot-Robot | `VLM-RR-Cent` | `VLM-RR-Dialog` |
| Baseline | Human-Robot | `VLM-HR-Cent` | `VLM-HR-Dialog` |
| Ours | Robot-Robot | `CRIE-BT-RR-Cent` | `CRIE-BT-RR-Dialog` |
| Ours | Human-Robot | `CRIE-BT-HR-Cent` | `CRIE-BT-HR-Dialog` |

## Implementation names

Use stable machine names without hyphens:

| Paper name | Code name |
|---|---|
| `VLM-RR-Cent` | `vlm_rr_cent` |
| `VLM-RR-Dialog` | `vlm_rr_dialog` |
| `VLM-HR-Cent` | `vlm_hr_cent` |
| `VLM-HR-Dialog` | `vlm_hr_dialog` |
| `CRIE-BT-RR-Cent` | `criebt_rr_cent` |
| `CRIE-BT-RR-Dialog` | `criebt_rr_dialog` |
| `CRIE-BT-HR-Cent` | `criebt_hr_cent` |
| `CRIE-BT-HR-Dialog` | `criebt_hr_dialog` |

## Stage-specific backend mapping

| Stage | Conditions included | Skill backend | Baseline monitor | Ours monitor | Environment |
|---|---|---|---|---|---|
| Step 1 | RR only | RRT | VLM self-monitor from image/status feedback | coded simulator/RRT monitor | simulation |
| Step 2 | HR only initially, optionally RR regression | RRT robot + terminal human | VLM self-monitor from image/status/dialogue feedback | coded simulator/RRT monitor | simulation |
| Step 3 | HR only | learned visual skills | VLM self-monitor from camera/status/dialogue feedback | SARM progress monitor | real world |

## Why not name the baseline SARM?

The baseline does not use a separate SARM monitor. It uses a monolithic VLM planner that must infer progress and failure from image feedback inside the same planning loop. SARM belongs to the CRIE-BT monitor backend in the real-world stage.

## Why keep CRIE-BT name constant?

CRIE-BT is the architecture. The monitor implementation changes by stage:

- `CRIE-BT + CodedSimMonitor` in simulation;
- `CRIE-BT + SARMMonitor` in real-world experiments.

This avoids renaming the method whenever the monitor backend changes.
