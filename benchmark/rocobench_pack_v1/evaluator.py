"""Phase 8 benchmark evaluator."""

import os
import shutil
from typing import Any, Dict, List, Mapping, Tuple

from .baselines import build_baseline
from .config import EvaluationRunConfig
from .report import write_run_outputs
from .results import BenchmarkEpisodeResult, write_episode_json
from .tasks import BenchmarkTask, load_tasks, tasks_by_id
from .variations import BenchmarkVariation, manifest_hash, select_variations
from .version import (
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    PREDICATE_VERSION,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    TASK_MANIFEST_PATH,
    VARIATION_MANIFEST_PATH,
    version_record,
)


class BenchmarkEvaluationError(RuntimeError):
    """Raised when a benchmark run cannot be completed."""


class BenchmarkEvaluator:
    def __init__(self, config: EvaluationRunConfig) -> None:
        self.config = config
        self.tasks = tasks_by_id(load_tasks())
        self.variation_manifest_hash = manifest_hash()

    def _task(self, task_id: str) -> BenchmarkTask:
        if task_id not in self.tasks:
            raise BenchmarkEvaluationError("unknown benchmark task: {}".format(task_id))
        return self.tasks[task_id]

    def _episode_result(
        self,
        task: BenchmarkTask,
        variation: BenchmarkVariation,
        episode_index: int,
        data: Mapping[str, Any],
    ) -> BenchmarkEpisodeResult:
        return BenchmarkEpisodeResult(
            benchmark_id=BENCHMARK_ID,
            benchmark_version=BENCHMARK_VERSION,
            schema_version=SCHEMA_VERSION,
            protocol_version=PROTOCOL_VERSION,
            predicate_version=PREDICATE_VERSION,
            variation_manifest_hash=self.variation_manifest_hash,
            task_id=task.task_id,
            variation_id=variation.variation_id,
            variation_hash=variation.variation_hash,
            track=self.config.track,
            method=self.config.method,
            episode_index=int(episode_index),
            seed=int(variation.seed),
            success=bool(data.get("success", False)),
            overall_success=bool(data.get("overall_success", data.get("success", False))),
            learned_success=bool(data.get("learned_success", False)),
            fallback_success=bool(data.get("fallback_success", False)),
            termination_reason=str(data.get("termination_reason", "")),
            num_env_steps=int(data.get("num_env_steps", 0)),
            sim_time_s=float(data.get("sim_time_s", 0.0)),
            latency_p50_ms=float(data.get("latency_p50_ms", 0.0)),
            latency_p95_ms=float(data.get("latency_p95_ms", 0.0)),
            action_violations=int(data.get("action_violations", 0)),
            drops=int(data.get("drops", 0)),
            collisions=int(data.get("collisions", 0)),
            final_error=float(data.get("final_error", 0.0)),
            plan_valid=bool(data.get("plan_valid", False)),
            replans=int(data.get("replans", 0)),
            fallback_attempts=int(data.get("fallback_attempts", 0)),
            planner_latency_ms=float(data.get("planner_latency_ms", 0.0)),
            token_count=int(data.get("token_count", 0)),
            makespan_steps=int(data.get("makespan_steps", 0)),
            parallel_speedup=float(data.get("parallel_speedup", 0.0)),
            resource_violations=int(data.get("resource_violations", 0)),
            central_stops=int(data.get("central_stops", 0)),
            artifact_dir=str(data.get("artifact_dir", "")),
            extra=dict(data.get("extra", {})),
        )

    def _run_manifest(self, results: Tuple[BenchmarkEpisodeResult, ...]) -> Dict[str, Any]:
        tasks = [self._task(task_id).to_dict() for task_id in self.config.resolved_task_ids]
        variations = [
            {
                "task_id": result.task_id,
                "variation_id": result.variation_id,
                "variation_hash": result.variation_hash,
                "seed": result.seed,
            }
            for result in results
        ]
        record = version_record()
        record.update(
            {
                "suite": self.config.suite,
                "track": self.config.track,
                "method": self.config.method,
                "task_manifest": os.path.relpath(TASK_MANIFEST_PATH, os.getcwd()),
                "variation_manifest": os.path.relpath(VARIATION_MANIFEST_PATH, os.getcwd()),
                "variation_manifest_hash": self.variation_manifest_hash,
                "config": self.config.to_dict(),
                "tasks": tasks,
                "episodes": variations,
            }
        )
        return record

    def evaluate(self) -> Tuple[Dict[str, Any], Tuple[BenchmarkEpisodeResult, ...], Dict[str, Any]]:
        output_dir = os.path.abspath(self.config.output_dir)
        if os.path.exists(output_dir):
            if not self.config.overwrite:
                raise BenchmarkEvaluationError("output directory exists; use --overwrite: {}".format(output_dir))
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        baseline = build_baseline(self.config)
        results: List[BenchmarkEpisodeResult] = []
        episode_index = 0
        for task_id in self.config.resolved_task_ids:
            task = self._task(task_id)
            variations = select_variations(task.variation_group, int(self.config.episodes_per_task))
            for variation in variations:
                episode_dir = os.path.join(
                    output_dir,
                    "episodes",
                    "episode_{:06d}_{}_{}".format(episode_index, task.task_id.replace(".", "_"), variation.variation_id),
                )
                data = baseline.run(task, variation, episode_index, episode_dir)
                result = self._episode_result(task, variation, episode_index, data)
                write_episode_json(os.path.join(episode_dir, "benchmark_result.json"), result)
                results.append(result)
                episode_index += 1
        results_tuple = tuple(results)
        run_manifest = self._run_manifest(results_tuple)
        metrics = write_run_outputs(output_dir, run_manifest, results_tuple)
        return run_manifest, results_tuple, metrics

