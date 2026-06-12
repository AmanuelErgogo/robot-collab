# Traceability and Gate Matrix

| Requirement | Phase | Artifact | Required verification |
|---|---:|---|---|
| Dataset loads through public LeRobotDataset | 2 | dataset manifest | loader integration |
| Observation precedes action | 2 | episode records | alignment and replay |
| No split leakage | 2 | splits.json | leakage test |
| Expert actions replay | 2 | replay report | simulator replay |
| ACT debug train | 3 | checkpoint | trainer integration |
| Tiny subset overfits | 3 | metrics | overfit test |
| Independent checkpoint reload | 3 | checkpoint metadata | clean process |
| Native-unit model action | 3/4 | processor trace | postprocess test |
| Direct rollout | 4 | result/video | simulator rollout |
| Chunk horizon correct | 4 | chunk trace | queue test |
| Success uses task predicate | 4/5 | result | false-success tests |
| Learned executor satisfies interface | 5 | executor | contract test |
| Structured failure evidence | 5 | result/events | monitor tests |
| Explicit fallback attribution | 5/6 | fallback event | accounting test |
| Current-state planner feedback | 6 | feedback/event log | stale-state negative |
| Bounded replanning | 6 | budget | exhaustion test |
| Rollback restores state | 5/6 | state digest | equality test |
| Multi-agent sequential execution | 7 | combined result | scheduler integration |
| One simulator step per joint tick | 7 | step trace | joint-step test |
| Resource/workspace conflict rejection | 7 | schedule | conflict tests |
| Severe event STOP_ALL | 7 | central event | cancellation test |
| Immutable benchmark variations | 8 | manifest hash | release validator |
| Same variations across methods | 8 | run manifests | paired-run check |
| Skill/planner tracks separate | 8 | result schema | schema test |
| LeRobot package entry works | 8 | package | clean install smoke |
| Raw episode metrics saved | 8 | episodes.csv | report validation |
| Complete version provenance | 2–8 | manifests | reproducibility audit |

# Gate 2→3
- [ ] loader passes
- [ ] replay passes
- [ ] splits frozen
- [ ] schema hash saved
- [ ] validator has no critical errors

# Gate 3→4
- [ ] debug train
- [ ] tiny overfit
- [ ] independent reload
- [ ] native postprocess
- [ ] test split untouched

# Gate 4→5
- [ ] direct rollout
- [ ] queue reset/horizon tests
- [ ] failure reasons reliable
- [ ] artifacts aligned

# Gate 5→6
- [ ] executor failure paths
- [ ] cancellation
- [ ] rollback
- [ ] fallback attribution
- [ ] RRT regression

# Gate 6→7
- [ ] canned planner scenarios
- [ ] current-state feedback
- [ ] bounded budgets
- [ ] loop detection
- [ ] sequential routing

# Gate 7→8
- [ ] joint step correctness
- [ ] conservative conflicts
- [ ] STOP_ALL
- [ ] sequential fallback
- [ ] sequential/concurrent comparison
