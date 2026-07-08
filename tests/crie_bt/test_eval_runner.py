import json

from scripts.run_crie_bt_eval import main as run_main
from scripts.analyze_crie_bt_eval import summarize, summarize_grouped


def test_eval_runner_writes_valid_jsonl(tmp_path):
    output = tmp_path / "eval.jsonl"
    assert run_main([
        "--task", "pack",
        "--mode", "all",
        "--episodes", "1",
        "--executor", "scripted",
        "--planner", "scripted",
        "--output", str(output),
    ]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    # open_loop was removed from the evaluated paper methods; --mode all now
    # covers the direct-feedback baseline, the bt-mediated method, and the
    # vlm/sarm monitor-planner.
    assert {row["mode"] for row in rows} == {
        "direct_feedback", "bt_mediated", "vlm_sarm_monitor_planner"}
    summaries = summarize(str(output))
    assert len(summaries) == 3


def test_analyzer_summarizes_paper_metrics(tmp_path):
    output = tmp_path / "paper_metrics.jsonl"
    rows = [
        {
            "task_id": "sandwich",
            "paper_method": "CRIE-BT-Cent",
            "mode": "bt_mediated",
            "success": True,
            "sim_success": True,
            "steps": 2,
            "planner_calls": 1,
            "replans": 0,
            "local_retries": 0,
            "failure_counts": {},
            "explanations": [],
            "wall_time_s": 4.0,
            "llm_call_latencies_s": [1.0, 3.0],
            "llm_prompt_tokens": 10,
            "llm_completion_tokens": 5,
            "llm_total_tokens": 15,
            "reactivity_s": 0.5,
            "hallucination_count": 1,
            "hallucination_annotation_count": 4,
        },
        {
            "task_id": "sandwich",
            "paper_method": "CRIE-BT-Cent",
            "mode": "bt_mediated",
            "success": False,
            "sim_success": False,
            "steps": 3,
            "planner_calls": 2,
            "replans": 1,
            "local_retries": 1,
            "failure_counts": {"MISSED_GRASP": 1},
            "explanations": ["retry"],
            "wall_time_s": 6.0,
            "llm_call_latencies_s": [2.0],
            "llm_prompt_tokens": 20,
            "llm_completion_tokens": 8,
            "llm_total_tokens": 28,
            "reactivity_s": 1.5,
            "hallucination_count": 0,
            "hallucination_annotation_count": 4,
        },
    ]
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summaries = summarize_grouped(str(output), group_by="task_method")
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["mode"] == "sandwich:CRIE-BT-Cent"
    assert summary["success_rate"] == 0.5
    assert summary["task_success_rate"] == 0.5
    assert summary["avg_wall_time_s"] == 5.0
    assert summary["avg_llm_latency_s"] == 2.0
    assert summary["avg_llm_total_tokens"] == 21.5
    assert summary["avg_reactivity_s"] == 1.0
    assert summary["hallucination_rate"] == 0.125
