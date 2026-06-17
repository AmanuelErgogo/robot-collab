"""Release validation for RoCoBench-Pack-Skills-v1."""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .baselines import BridgeStepBaseline
from .config import EvaluationRunConfig, METHOD_HOLD
from .processors import BenchmarkObservationProcessor
from .report import write_json
from .results import EPISODE_CSV_FIELDS
from .tasks import load_task_manifest, load_tasks
from .variations import load_variation_manifest, load_variations, manifest_hash
from .version import (
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    COMPATIBILITY_LOCK_PATH,
    REPO_ROOT,
    SCHEMA_MANIFEST_PATH,
    TASK_MANIFEST_PATH,
    VARIATION_MANIFEST_PATH,
    read_json,
)


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "message": self.message}


class ReleaseValidator:
    def __init__(self, run_bridge_smoke: bool = False, require_bridge: bool = False) -> None:
        self.run_bridge_smoke = bool(run_bridge_smoke)
        self.require_bridge = bool(require_bridge)
        self.checks: List[ValidationCheck] = []

    def pass_(self, name: str, message: str = "") -> None:
        self.checks.append(ValidationCheck(name, "pass", message))

    def fail(self, name: str, message: str) -> None:
        self.checks.append(ValidationCheck(name, "fail", message))

    def block(self, name: str, message: str) -> None:
        self.checks.append(ValidationCheck(name, "blocked", message))

    def _check(self, name: str, fn) -> None:
        try:
            message = fn() or ""
        except Exception as exc:
            self.fail(name, str(exc))
            return
        self.pass_(name, str(message))

    def validate(self) -> Dict[str, Any]:
        self._check("package_import", self._package_import)
        self._check("manifests_load_and_hash", self._manifests_load_and_hash)
        self._check("task_variation_coverage", self._task_variation_coverage)
        self._check("result_schema", self._result_schema)
        self._check("processor_contract", self._processor_contract)
        self._check("baseline_schema_smoke", self._baseline_schema_smoke)
        self._check("docs_cards_license", self._docs_cards_license)
        self._check("provenance_lock", self._provenance_lock)
        self._check("no_phase8_secrets_or_private_paths", self._no_phase8_secrets_or_private_paths)
        self._bridge_smoke()
        failed = [check for check in self.checks if check.status == "fail"]
        blocked = [check for check in self.checks if check.status == "blocked"]
        return {
            "benchmark_id": BENCHMARK_ID,
            "benchmark_version": BENCHMARK_VERSION,
            "status": "fail" if failed else "pass",
            "failed": [check.to_dict() for check in failed],
            "blocked": [check.to_dict() for check in blocked],
            "checks": [check.to_dict() for check in self.checks],
        }

    def _package_import(self) -> str:
        import benchmark.rocobench_pack_v1 as package

        if package.BENCHMARK_ID != BENCHMARK_ID:
            raise AssertionError("benchmark ID mismatch")
        return package.BENCHMARK_VERSION

    def _manifests_load_and_hash(self) -> str:
        task_manifest = load_task_manifest()
        variation_manifest = load_variation_manifest()
        if task_manifest.get("benchmark_id") != BENCHMARK_ID:
            raise AssertionError("task manifest benchmark_id mismatch")
        if variation_manifest.get("benchmark_id") != BENCHMARK_ID:
            raise AssertionError("variation manifest benchmark_id mismatch")
        hash_value = manifest_hash()
        if len(hash_value) != 64:
            raise AssertionError("invalid variation hash")
        return hash_value

    def _task_variation_coverage(self) -> str:
        tasks = load_tasks()
        variations = load_variations()
        groups = {variation.variation_group for variation in variations}
        missing = [task.task_id for task in tasks if task.variation_group not in groups]
        if missing:
            raise AssertionError("tasks missing variations: {}".format(missing))
        for variation in variations:
            if not variation.expected_skill_plan:
                raise AssertionError("{} has no expected_skill_plan".format(variation.variation_id))
        return "{} tasks, {} variations".format(len(tasks), len(variations))

    def _result_schema(self) -> str:
        schema = read_json(SCHEMA_MANIFEST_PATH)
        missing = [field for field in schema.get("required_episode_fields", ()) if field not in EPISODE_CSV_FIELDS]
        if missing:
            raise AssertionError("required schema fields missing from CSV: {}".format(missing))
        return "episode CSV fields={}".format(len(EPISODE_CSV_FIELDS))

    def _processor_contract(self) -> str:
        processor = BenchmarkObservationProcessor(required_cameras=("front", "active_agent"))
        obs = {
            "agent_pos": np.zeros((2,), dtype=np.float32),
            "pixels": {
                "front": np.zeros((4, 5, 3), dtype=np.uint8),
                "active_agent": np.zeros((4, 5, 3), dtype=np.uint8),
            },
        }
        processed = processor.process(obs)
        if processed["observation.images.front"].dtype != np.uint8:
            raise AssertionError("processor normalized image data")
        if processor.last_trace is None:
            raise AssertionError("processor did not record trace")
        return "processor maps raw keys only"

    def _baseline_schema_smoke(self) -> str:
        class ActionSpace:
            low = np.asarray([-1.0, -1.0], dtype=np.float32)
            high = np.asarray([1.0, 1.0], dtype=np.float32)
            shape = (2,)

        class FakeEnv:
            metadata = {"render_fps": 10}
            action_space = ActionSpace()

            def __init__(self):
                self.steps = 0

            def hold_action(self):
                return np.zeros((2,), dtype=np.float32)

            def reset(self, seed=None):
                del seed
                self.steps = 0
                return {}, {"is_success": False}

            def step(self, action):
                del action
                self.steps += 1
                return {}, float(self.steps >= 1), self.steps >= 1, False, {"is_success": self.steps >= 1}

            def close(self):
                return None

        from .tasks import BenchmarkTask
        from .variations import BenchmarkVariation

        task = BenchmarkTask(
            task_id="pack.put.alice",
            description="fake",
            active_agents=("Alice",),
            max_steps=2,
            success_predicate="fake",
            failure_predicates=("max_steps",),
            action_representation="absolute_joint_position_plus_gripper",
            required_cameras=("front",),
            variation_group="fake",
        )
        variation = BenchmarkVariation(
            variation_id="fake-var",
            variation_group="fake",
            seed=1,
            active_agents=("Alice",),
            object_assignments={"Alice": "apple"},
            target_slots={"Alice": "bin_front_left"},
            distractors=(),
            pose_set="fake",
            concurrency_case="single_agent",
            expected_skill_plan=({"agent_name": "Alice", "skill_name": "PUT_OBJECT_IN_CONTAINER", "object": "apple", "container": "bin_front_left"},),
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = EvaluationRunConfig(method=METHOD_HOLD, output_dir=tmp, require_bridge=False)
            baseline = BridgeStepBaseline(METHOD_HOLD, config, env_factory=lambda cfg: FakeEnv())
            result = baseline.run(task, variation, 0, os.path.join(tmp, "episode"))
        if not result["success"]:
            raise AssertionError("fake hold baseline did not produce success schema")
        return "hold baseline schema smoke passed"

    def _docs_cards_license(self) -> str:
        required = [
            "benchmark/rocobench_pack_v1/README.md",
            "benchmark/rocobench_pack_v1/LICENSE",
            "benchmark/rocobench_pack_v1/pyproject.toml",
            "benchmark/cards/environment_card.md",
            "benchmark/cards/dataset_card.md",
            "benchmark/cards/model_card_template.md",
            "docs/phase8_benchmark_release.md",
        ]
        missing = [path for path in required if not os.path.exists(os.path.join(REPO_ROOT, path))]
        if missing:
            raise AssertionError("missing docs/cards/license: {}".format(missing))
        return "{} files".format(len(required))

    def _provenance_lock(self) -> str:
        if not os.path.exists(COMPATIBILITY_LOCK_PATH):
            raise AssertionError("missing compatibility lock")
        lock = read_json(COMPATIBILITY_LOCK_PATH)
        required = ["lerobot_version", "lerobot_commit", "bridge_protocol", "schema_hash", "action_representation"]
        missing = [key for key in required if not lock.get(key)]
        if missing:
            raise AssertionError("compatibility lock missing keys: {}".format(missing))
        return "LeRobot {} {}".format(lock.get("lerobot_version"), lock.get("lerobot_commit"))

    def _no_phase8_secrets_or_private_paths(self) -> str:
        scanned = [
            "benchmark/rocobench_pack_v1",
            "benchmark/manifests",
            "benchmark/cards",
            "docs/phase8_benchmark_release.md",
            "scripts/evaluate_rocobench.py",
            "scripts/validate_rocobench_release.py",
            "scripts/compare_rocobench_runs.py",
            "scripts/generate_rocobench_report.py",
        ]
        patterns = [
            re.compile(r"sk-[A-Za-z0-9_\\-]{16,}"),
            re.compile("/" + "home" + r"/[^\\s`'\"]+"),
            re.compile("~/" + ".secrets"),
            re.compile("private " + "checkpoint", re.IGNORECASE),
        ]
        offenders: List[str] = []
        for rel in scanned:
            path = os.path.join(REPO_ROOT, rel)
            if os.path.isdir(path):
                for root, _dirs, files in os.walk(path):
                    for name in files:
                        offenders.extend(self._scan_file(os.path.join(root, name), patterns))
            elif os.path.exists(path):
                offenders.extend(self._scan_file(path, patterns))
        if offenders:
            raise AssertionError("sensitive/private references found: {}".format(offenders[:5]))
        return "phase8 files scanned"

    def _scan_file(self, path: str, patterns: List[Any]) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            return []
        matches = []
        for pattern in patterns:
            if pattern.search(text):
                matches.append(os.path.relpath(path, REPO_ROOT))
        return matches

    def _bridge_smoke(self) -> None:
        if not self.run_bridge_smoke:
            self.block("bridge_reset_step_render", "not run; pass --run-bridge-smoke to require a live Phase 0 server")
            return
        try:
            from .env import make_env

            env = make_env(
                cfg={
                    "endpoint": "tcp://127.0.0.1:5557",
                    "active_agent": "Alice",
                    "max_episode_steps": 2,
                    "request_timeout_ms": 30000,
                }
            )
            try:
                _obs, _info = env.reset(seed=8100)
                _obs, _reward, _terminated, _truncated, _info = env.step(env.hold_action())
                frame = env.render()
                if frame is None or getattr(frame, "ndim", 0) != 3:
                    raise AssertionError("render did not return HWC image")
            finally:
                env.close()
        except Exception as exc:
            if self.require_bridge:
                self.fail("bridge_reset_step_render", str(exc))
            else:
                self.block("bridge_reset_step_render", str(exc))
            return
        self.pass_("bridge_reset_step_render", "live bridge reset/step/render passed")


def validate_release(run_bridge_smoke: bool = False, require_bridge: bool = False, output_path: Optional[str] = None) -> Dict[str, Any]:
    result = ReleaseValidator(run_bridge_smoke=run_bridge_smoke, require_bridge=require_bridge).validate()
    if output_path:
        write_json(output_path, result)
    return result
