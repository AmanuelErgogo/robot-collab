# 12 — Risk Register

| Risk | Severity | Mitigation |
|---|---:|---|
| Project becomes too broad | High | Start with one domain, 5–8 skills, 3 planners, 2 robot backends, 1 human model |
| Learned skills fail often | High | Keep RRT expert, use fallback, report learned success separately |
| Planner evaluation confounded by execution failure | High | Provide planner-only and planner+execution tracks |
| Human model unrealistic | Medium | Start simple, report limitations, later validate with real users |
| Concurrency unsafe | High | Default sequential, require resource/workspace proofs, STOP_ALL |
| LeRobot/RoCo dependency mismatch | High | Maintain two runtimes and version locks |
| Metrics too many or unclear | Medium | Define primary metrics first: success, recovery, redundant work, idle time, safety |
| Paper contribution unclear | High | Frame around skill-centric planner benchmarking, not only learned execution |
| Reproducibility weak | High | Use manifests, seeds, schema hashes, raw episode CSV |
| LLM tests nondeterministic | Medium | Use canned planner outputs for CI; live LLM only optional |
