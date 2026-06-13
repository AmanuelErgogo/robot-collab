# 09 — Related Work Matrix

## RoCo / RoCoBench
RoCo introduced LLM-based dialectic multi-robot collaboration and RoCoBench, a six-task benchmark for multi-robot collaboration. It combines LLM discussion, structured plans, waypoint/path proposals, geometric feedback, and multi-arm motion planning.

SkillCoord-HRC difference: typed skills, learned/classical/human/scripted backends, human agents as first-class entities, planner benchmark tracks, and failure recovery metrics.

## PARTNR
PARTNR benchmarks planning and reasoning for human–robot coordination in household activities with heterogeneous capabilities, spatial constraints, and temporal constraints.

SkillCoord-HRC difference: focuses on executable tabletop/robot-skill coordination and backend-swappable skills with low-level execution monitoring.

## LeRobot
LeRobot provides infrastructure for robot datasets, policy training, and evaluation, including ACT and VLA models.

SkillCoord-HRC difference: uses LeRobot as a learned-skill backend while adding multi-agent coordination, human modeling, scheduling, failure-aware feedback, and HRC metrics.

## Tool-RoCo
Tool-RoCo extends RoCo-style multi-robot cooperation to tool/agent selection paradigms and evaluates centralized/decentralized cooperation patterns.

SkillCoord-HRC difference: focuses on executable human/robot skills, backend-swappable control, learned policy execution, failure injection, and HRC metrics.

## Related-work comparison table

| System | Multi-agent planning | Human agent | Learned skill backend | Failure recovery | Planner benchmark | HRC metrics |
|---|---:|---:|---:|---:|---:|---:|
| RoCoBench | Yes | Limited | Not primary | Geometric feedback | Yes | Limited |
| PARTNR | Yes | Yes | Abstract/varied | Planning focus | Yes | Some |
| LeRobot | No | No | Yes | Policy-eval focus | No | No |
| Tool-RoCo | Yes | Limited | Not primary | Tool feedback | Yes | Limited |
| SkillCoord-HRC | Yes | Yes | Yes | Yes | Yes | Yes |

## BibTeX starter

```bibtex
@misc{zhao2023roco,
  title={RoCo: Dialectic Multi-Robot Collaboration with Large Language Models},
  author={Zhao, Mandi and Jain, Shreeya and Song, Shuran},
  year={2023},
  eprint={2307.04738},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}

@misc{chang2024partnr,
  title={PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks},
  author={Chang, Matthew and others},
  year={2024},
  eprint={2411.00081},
  archivePrefix={arXiv}
}

@misc{zhang2025toolroco,
  title={Tool-RoCo: An Agent-as-Tool Self-organization Large Language Model Benchmark in Multi-robot Cooperation},
  author={Zhang, Ke and others},
  year={2025},
  eprint={2511.21510},
  archivePrefix={arXiv}
}
```
