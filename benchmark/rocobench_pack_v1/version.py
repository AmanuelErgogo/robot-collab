"""Benchmark version and provenance helpers."""

import json
import os
import subprocess
from typing import Any, Dict, Optional


BENCHMARK_ID = "RoCoBench-Pack-Skills-v1"
BENCHMARK_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
PROTOCOL_VERSION = "phase8.benchmark.v1"
PREDICATE_VERSION = "pack.predicate.v1"
SUITE_NAME = "rocobench_pack_skills_v1"

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
MANIFEST_DIR = os.path.join(REPO_ROOT, "benchmark", "manifests")
TASK_MANIFEST_PATH = os.path.join(MANIFEST_DIR, "pack_v1_tasks.json")
VARIATION_MANIFEST_PATH = os.path.join(MANIFEST_DIR, "pack_v1_variations.json")
SCHEMA_MANIFEST_PATH = os.path.join(MANIFEST_DIR, "pack_v1_schema.json")
COMPATIBILITY_LOCK_PATH = os.path.join(REPO_ROOT, "integrations", "lerobot_roco", "compatibility.lock.json")


def _git(args):
    try:
        return subprocess.check_output(["git"] + list(args), cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def git_commit() -> Optional[str]:
    return _git(["rev-parse", "HEAD"])


def git_dirty() -> Optional[bool]:
    status = _git(["status", "--short"])
    if status is None:
        return None
    return bool(status.strip())


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return data


def compatibility_lock() -> Dict[str, Any]:
    if not os.path.exists(COMPATIBILITY_LOCK_PATH):
        return {}
    return read_json(COMPATIBILITY_LOCK_PATH)


def version_record() -> Dict[str, Any]:
    lock = compatibility_lock()
    return {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "predicate_version": PREDICATE_VERSION,
        "roco_commit": git_commit(),
        "roco_dirty": git_dirty(),
        "lerobot_version": lock.get("lerobot_version"),
        "lerobot_commit": lock.get("lerobot_commit"),
        "bridge_protocol": lock.get("bridge_protocol"),
        "action_representation": lock.get("action_representation"),
        "compatibility_lock": os.path.relpath(COMPATIBILITY_LOCK_PATH, REPO_ROOT),
    }

