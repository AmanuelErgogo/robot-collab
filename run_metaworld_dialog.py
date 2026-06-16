"""Run a prompt-based controller on MetaWorld tasks.

The default mode is `expert`, which uses MetaWorld's scripted policy as a smoke
test for the bridge. `llm` mode queries the repo's LLM client for direct 4D
Sawyer control commands.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

from llm_api import create_llm_client
from metaworld_integration import (
    MetaWorldActionParser,
    MetaWorldBridgeClient,
    MetaWorldPromptEnv,
    resolve_task_spec,
)


def _manual_action():
    raw = input("Enter ACTION [dx, dy, dz, gripper]: ").strip()
    parser = MetaWorldActionParser()
    success, reason, action = parser.parse("EXECUTE\n" + raw if "EXECUTE" not in raw else raw)
    if not success:
        raise ValueError(reason)
    return action, raw


def _query_llm(llm_client, prompt_env, parser, snapshot, feedback_text, max_replans, temperature):
    system_prompt = prompt_env.build_system_prompt()
    last_error = None
    for _ in range(max_replans):
        user_prompt = prompt_env.build_user_prompt(snapshot, feedback_text=feedback_text)
        llm_response = llm_client.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=temperature,
        )
        success, reason, action = parser.parse(llm_response.text)
        if success:
            return action, llm_response.text, llm_response.usage, user_prompt
        last_error = "{}\n{}".format(reason, llm_response.text)
        feedback_text = "The previous response was invalid: {}".format(reason)
    raise RuntimeError("LLM failed to produce a valid MetaWorld action: {}".format(last_error))


def run_episode(
    task_name,
    args,
    episode_index,
):
    with MetaWorldBridgeClient(
        task_name=task_name,
        python_bin=args.metaworld_python_bin,
        repo_dir=args.metaworld_repo_dir,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
    ) as bridge:
        prompt_env = MetaWorldPromptEnv(bridge, max_steps=args.max_steps)
        parser = MetaWorldActionParser()
        llm_client = None
        if args.control_mode == "llm":
            llm_client = create_llm_client(args.llm_source, api_key_path=args.api_key_path)

        episode_dir = os.path.join(
            args.data_dir,
            args.run_name,
            task_name,
            "episode_{:03d}".format(episode_index),
        )
        os.makedirs(episode_dir, exist_ok=True)
        frames_dir = os.path.join(episode_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        jsonl_path = os.path.join(episode_dir, "lerobot_steps.jsonl")

        seed = args.seed + episode_index if args.seed is not None else None
        snapshot = prompt_env.reset(seed=seed)
        feedback_text = ""
        final_success = False

        with open(jsonl_path, "w") as record_file:
            for step_idx in range(args.max_steps):
                image_path = None
                if args.save_images:
                    image_path = os.path.join(frames_dir, "step_{:04d}.png".format(step_idx))
                    prompt_env.save_frame(image_path)

                if args.control_mode == "expert":
                    action = prompt_env.expert_action()
                    response_text = parser.format_action(action)
                    usage = {}
                    prompt_text = prompt_env.describe_state(snapshot)
                elif args.control_mode == "manual":
                    action, raw_text = _manual_action()
                    response_text = "EXECUTE\n" + raw_text if "EXECUTE" not in raw_text else raw_text
                    usage = {}
                    prompt_text = prompt_env.describe_state(snapshot)
                else:
                    action, response_text, usage, prompt_text = _query_llm(
                        llm_client=llm_client,
                        prompt_env=prompt_env,
                        parser=parser,
                        snapshot=snapshot,
                        feedback_text=feedback_text,
                        max_replans=args.num_replans,
                        temperature=args.temperature,
                    )

                outcome = prompt_env.step(action)
                feedback_text = prompt_env.build_feedback(snapshot, action, outcome)
                record = prompt_env.to_lerobot_record(
                    episode_index=episode_index,
                    step_index=step_idx,
                    snapshot=snapshot,
                    action=action,
                    outcome_snapshot=outcome,
                    image_path=image_path,
                    prompt_text=prompt_text,
                    response_text=response_text,
                )
                record["usage"] = usage
                record_file.write(json.dumps(record) + "\n")
                record_file.flush()

                step_debug_path = os.path.join(episode_dir, "step_{:04d}.json".format(step_idx))
                with open(step_debug_path, "w") as step_file:
                    json.dump(
                        {
                            "snapshot": snapshot,
                            "response": response_text,
                            "action": np.asarray(action).tolist(),
                            "outcome": outcome,
                            "feedback": feedback_text,
                            "usage": usage,
                        },
                        step_file,
                        indent=2,
                    )

                print(
                    "[{}] step={} reward={} success={}".format(
                        task_name,
                        step_idx,
                        outcome.get("reward"),
                        outcome.get("success"),
                    )
                )

                if outcome.get("success") or outcome.get("terminated") or outcome.get("truncated"):
                    final_success = bool(outcome.get("success"))
                    break
                snapshot = outcome

        return final_success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--control_mode", choices=["expert", "llm", "manual"], default="expert")
    parser.add_argument("--llm_source", default="gpt-4")
    parser.add_argument("--api_key_path", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_replans", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--metaworld_python_bin", default=None)
    parser.add_argument("--metaworld_repo_dir", default=None)
    parser.add_argument("--camera_name", default="corner2")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = "metaworld_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    tasks = resolve_task_spec(args.task)
    os.makedirs(os.path.join(args.data_dir, args.run_name), exist_ok=True)
    with open(os.path.join(args.data_dir, args.run_name, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    results = {}
    for task_name in tasks:
        task_successes = []
        for episode_index in range(args.episodes_per_task):
            print("=== task={} episode={} mode={} ===".format(task_name, episode_index, args.control_mode))
            task_successes.append(run_episode(task_name, args, episode_index))
        results[task_name] = task_successes

    summary_path = os.path.join(args.data_dir, args.run_name, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print("Saved MetaWorld run summary to {}".format(summary_path))


if __name__ == "__main__":
    main()
