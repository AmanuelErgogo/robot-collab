"""Markdown report helpers for Phase 3 runs."""

import os
from typing import Any, Mapping, Optional

from integrations.lerobot_roco.dataset.manifest import read_json


def _code(value: Any) -> str:
    return "`{}`".format(value)


def render_phase3_report(manifest: Mapping[str, Any], checkpoint: Optional[Mapping[str, Any]] = None) -> str:
    preflight = manifest.get("preflight", {})
    compatibility = manifest.get("compatibility", {})
    issues = preflight.get("issues", [])
    lines = [
        "# Phase 3 ACT Training Report",
        "",
        "## Run",
        "- run_id: {}".format(_code(manifest.get("run_id"))),
        "- status: {}".format(_code(manifest.get("status"))),
        "- phase: {}".format(_code(manifest.get("phase"))),
        "- resolved_config_hash: {}".format(_code(manifest.get("resolved_config_hash"))),
        "",
        "## Dataset",
    ]
    dataset = manifest.get("dataset", {})
    for key in ("dataset_root", "dataset_revision", "schema_hash", "bridge_protocol", "action_representation", "fps"):
        lines.append("- {}: {}".format(key, _code(dataset.get(key))))
    lines.extend(["", "## Preflight"])
    lines.append("- ok: {}".format(_code(preflight.get("ok"))))
    if not issues:
        lines.append("- issues: `none`")
    else:
        for issue in issues:
            lines.append("- [{}] {}: {}".format(issue.get("severity"), issue.get("code"), issue.get("message")))
    lines.extend(["", "## LeRobot Environment"])
    lines.append("- ok: {}".format(_code(compatibility.get("ok"))))
    lines.append("- lerobot_version: {}".format(_code(compatibility.get("lerobot_version"))))
    lines.append("- lerobot_commit: {}".format(_code(compatibility.get("lerobot_commit"))))
    for error in compatibility.get("errors", []):
        lines.append("- error: {}".format(error))
    for warning in compatibility.get("warnings", []):
        lines.append("- warning: {}".format(warning))
    lines.extend(["", "## Command"])
    lines.append("```bash")
    lines.append(" \\\n  ".join(manifest.get("command", [])))
    lines.append("```")
    if checkpoint is not None:
        lines.extend(["", "## Checkpoint"])
        lines.append("- ok: {}".format(_code(checkpoint.get("ok"))))
        lines.append("- pretrained_model_dir: {}".format(_code(checkpoint.get("pretrained_model_dir"))))
        for issue in checkpoint.get("issues", []):
            lines.append("- [{}] {}: {}".format(issue.get("severity"), issue.get("code"), issue.get("message")))
    lines.extend(
        [
            "",
            "## Notes",
            "- Offline training diagnostics are not manipulation success.",
            "- Closed-loop simulator rollout is Phase 4.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase3_report(run_dir: str, checkpoint: Optional[Mapping[str, Any]] = None) -> str:
    manifest = read_json(os.path.join(run_dir, "run_manifest.json"))
    path = os.path.join(run_dir, "phase3_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_phase3_report(manifest, checkpoint=checkpoint))
    return path
