import json

from scripts.run_crie_bt_eval import main as run_main
from scripts.analyze_crie_bt_eval import summarize


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
    assert {row["mode"] for row in rows} == {"open_loop", "direct_feedback", "bt_mediated"}
    summaries = summarize(str(output))
    assert len(summaries) == 3
