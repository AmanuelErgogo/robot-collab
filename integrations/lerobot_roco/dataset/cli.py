"""Command helpers for Phase 2 scripts."""

import argparse
import os
from typing import Any, Optional

from integrations.lerobot_roco.roco_runtime.config import RoCoBridgeServerConfig
from integrations.lerobot_roco.roco_runtime.server import RoCoBridgeServer

from .config import DatasetCollectionConfig
from .episode_sampler import DeterministicVariationSampler, reset_env_for_variation, with_observed_poses
from .expert_source import (
    build_episode_metadata,
    build_rrt_executor,
    build_skill_plan,
    check_put_object_postcondition,
    prepare_rrt_plan,
)
from .manifest import atomic_write_json, write_compatibility_lock
from .recorder import RocoTransitionObserver, annotate_record_stages
from .schema import build_schema_from_env_spec
from .splitter import create_variation_group_splits
from .statistics import compute_dataset_statistics
from .validator import validate_dataset
from .writer import AtomicEpisodeWriter


def create_env_from_config(config: DatasetCollectionConfig, seed: int) -> Any:
    server_config = RoCoBridgeServerConfig(
        task=config.task_id,
        active_agent=config.active_agent,
        seed=int(seed),
        image_height=config.image_height,
        image_width=config.image_width,
        camera_aliases=config.camera_aliases,
        max_episode_steps=config.max_episode_steps,
        headless=True,
    )
    server = RoCoBridgeServer(server_config)
    server.env.randomize_init = bool(config.randomize_init)
    server.env.render_point_cloud = bool(config.render_point_cloud)
    server._rebuild_adapters()
    return server.env, server


def collect_dataset(args: argparse.Namespace) -> int:
    config = DatasetCollectionConfig.from_yaml(args.config).with_overrides(
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    sampler = DeterministicVariationSampler(
        master_seed=args.master_seed,
        object_names=config.object_names,
        target_names=config.target_names,
        agent_name=config.active_agent,
    )
    env, server = create_env_from_config(config, args.master_seed)
    schema = build_schema_from_env_spec(server.spec, skill_id=config.skill_id)
    if config.fps is not None:
        schema_data = schema.to_dict()
        schema_data["fps"] = float(config.fps)
        from .schema import SkillDataSchema

        schema = SkillDataSchema.from_dict(schema_data)
    writer = AtomicEpisodeWriter(
        config.resolved_output_root(args.output_root),
        schema,
        resume=config.resume,
        overwrite=config.overwrite or args.overwrite,
    )
    write_compatibility_lock(
        os.path.join("integrations", "lerobot_roco", "compatibility.lock.json"),
        schema,
        repo_root=os.getcwd(),
    )

    start_index = int(args.start_index)
    for offset in range(int(args.num_episodes)):
        episode_index = start_index + offset
        variation = sampler.sample(episode_index)
        episode_id = "episode_{:06d}".format(episode_index)
        try:
            obs = reset_env_for_variation(env, variation)
            variation = with_observed_poses(env, variation, config.object_names)
            if writer.is_variation_committed(variation.variation_id) and config.resume and not args.overwrite:
                print(
                    "episode={} variation={} status=skipped reason=variation_already_committed".format(
                        episode_id,
                        variation.variation_id,
                    )
                )
                continue
            plan = build_skill_plan(
                config.active_agent,
                variation.object_name,
                variation.target_name,
                agent_names=(config.active_agent,) + tuple(config.passive_agents),
            )
            motion_target_name = str(config.expert_place_target_overrides.get(variation.target_name, variation.target_name))
            execution_plan = plan
            if motion_target_name != variation.target_name:
                execution_plan = build_skill_plan(
                    config.active_agent,
                    variation.object_name,
                    motion_target_name,
                    agent_names=(config.active_agent,) + tuple(config.passive_agents),
                )
            execution_plan = prepare_rrt_plan(env, obs, execution_plan)
            observer = RocoTransitionObserver(
                env,
                schema,
                active_agent=config.active_agent,
                episode_index=episode_index,
                camera_aliases=config.camera_aliases,
                image_height=config.image_height,
                image_width=config.image_width,
            )
            executor = build_rrt_executor(env, transition_observer=observer, max_sim_steps=config.max_episode_steps)
            result = executor.execute(execution_plan, obs)
            final_obs = observer.after_step_results[-1].get("observation") if observer.after_step_results else None
            del final_obs
            latest_obs = env.get_obs() if hasattr(env, "get_obs") else obs
            postcondition = check_put_object_postcondition(env, latest_obs, variation.object_name, variation.target_name)
            robot_name = getattr(env, "robot_name_map_inv", {}).get(config.active_agent)
            metadata = dict(
                build_episode_metadata(
                    config,
                    episode_id,
                    variation,
                    plan,
                    result,
                    postcondition,
                    robot_name,
                )
            )
            metadata["frame_count"] = len(observer.frames)
            metadata["expert_motion_target_name"] = motion_target_name
            metadata["expert_requested_target_name"] = variation.target_name
            metadata["expert_motion_target_overridden"] = bool(motion_target_name != variation.target_name)
            record = observer.to_episode_record(episode_id, variation, metadata)
            annotate_record_stages(record)
            if result.success and postcondition and record.frame_count >= config.min_episode_frames:
                status, written_episode_id = writer.write_episode(record)
                print(
                    "episode={} variation={} status={} committed_episode={} frames={}".format(
                        episode_id,
                        variation.variation_id,
                        status,
                        written_episode_id,
                        record.frame_count,
                    )
                )
            else:
                code = "POSTCONDITION_FAILED"
                if not result.success:
                    code = "EXPERT_EXECUTION_FAILED"
                quarantine_path = os.path.join(writer.dataset_root, "quarantine", episode_id)
                existed = os.path.exists(quarantine_path) and not writer.overwrite
                writer.quarantine_episode(episode_id, code, metadata.get("termination_reason", code), metadata=metadata, record=record)
                print(
                    "episode={} variation={} status={} code={} frames={} executor_success={} postcondition_success={}".format(
                        episode_id,
                        variation.variation_id,
                        "kept_existing_quarantine" if existed else "quarantined",
                        code,
                        record.frame_count,
                        bool(result.success),
                        bool(postcondition),
                    )
                )
        except KeyboardInterrupt:
            quarantine_path = os.path.join(writer.dataset_root, "quarantine", episode_id)
            existed = os.path.exists(quarantine_path) and not writer.overwrite
            writer.quarantine_episode(episode_id, "INTERRUPTED", "collection interrupted")
            print(
                "episode={} status={} code=INTERRUPTED".format(
                    episode_id,
                    "kept_existing_quarantine" if existed else "quarantined",
                )
            )
            raise
        except Exception as exc:
            quarantine_path = os.path.join(writer.dataset_root, "quarantine", episode_id)
            existed = os.path.exists(quarantine_path) and not writer.overwrite
            writer.quarantine_episode(episode_id, "EXPERT_PLANNING_FAILED", str(exc))
            print(
                "episode={} status={} code=EXPERT_PLANNING_FAILED reason={}".format(
                    episode_id,
                    "kept_existing_quarantine" if existed else "quarantined",
                    str(exc),
                )
            )
    return 0


def validate_dataset_cli(args: argparse.Namespace) -> int:
    report = validate_dataset(
        args.dataset_root,
        report_dir=args.report_dir,
        require_lerobot=args.require_lerobot,
    )
    print(report.to_markdown())
    return 0 if report.ok else 1


def create_splits_cli(args: argparse.Namespace) -> int:
    ratios = {"train": args.train, "validation": args.validation, "test": args.test}
    manifest = create_variation_group_splits(args.dataset_root, ratios=ratios, seed=args.seed)
    print(manifest.to_dict())
    return 0


def visualize_episode_cli(args: argparse.Namespace) -> int:
    from .writer import load_episode_arrays

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for visualization") from exc
    episode_path = os.path.join(args.dataset_root, "episodes", args.episode_id)
    arrays = load_episode_arrays(episode_path)
    os.makedirs(args.output, exist_ok=True)
    for key, value in arrays.items():
        if not key.startswith("images__"):
            continue
        alias = key.replace("images__", "")
        image = value[0]
        Image.fromarray(image).save(os.path.join(args.output, "{}_first.png".format(alias)))
        Image.fromarray(value[-1]).save(os.path.join(args.output, "{}_last.png".format(alias)))
    return 0


def stats_cli(args: argparse.Namespace) -> int:
    stats = compute_dataset_statistics(args.dataset_root)
    output = args.output or os.path.join(args.dataset_root, "statistics.json")
    atomic_write_json(output, stats)
    print(stats)
    return 0


def add_common_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)


def build_collect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_dataset_arg(parser)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--require-lerobot", action="store_true")
    return parser


def build_split_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_dataset_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    return parser


def build_visualize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_dataset_arg(parser)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", required=True)
    return parser
