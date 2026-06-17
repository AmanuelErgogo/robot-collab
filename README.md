# RoCo: Dialectic Multi-Robot Collaboration with Large Language Models
Codebase for paper: RoCo: Dialectic Multi-Robot Collaboration with Large Language Models

[Mandi Zhao](https://mandizhao.github.io), [Shreeya Jain](https://www.linkedin.com), [Shuran Song](https://www.cs.columbia.edu/~shurans/) 

[Arxiv](https://arxiv.org/abs/2307.04738) | [Project Website](https://project-roco.github.io) 

 
<img src="method.jpeg" alt="method" width="800"/>


## Setup
### RoCo simulator environment
```
conda create -n roco python=3.8 
conda activate roco
```
### Install MuJoCo and dm_control
```
python -m pip install --upgrade pip
pip install mujoco==2.3.0
pip install dm_control==1.0.8 
```
**If you have an M1 MacBook and would like to visualize the task scenes locally:**

Download the macOS-compatible `.dmg` file from the [MuJoCo release page](https://github.com/deepmind/mujoco/releases). Inside it should have a `MuJoCo.app` file that you can drag into your `/Applications` folder so it behaves like other macOS apps. You can then open the app and drag XML files into it. Find more information in the [official documentation](https://mujoco.readthedocs.io/en/latest/programming/#getting-started).

### Install repo packages
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

If you plan to run the Phase 0 bridge server and its smoke tests from the
Python 3.8 runtime, also install:
```
pip install -r requirements-phase0-roco.txt
```

On Linux headless machines, set:
```
export MUJOCO_GL=egl
```

### Split LeRobot environment for Phases 0, 3, 4, and 5
The bridge client, LeRobot dataset export, ACT training, and direct policy
rollout are designed to run in a separate environment from the Python 3.8
simulator runtime.

If you want the isolated packaged client, create a dedicated Python 3.12+
environment:
```
conda create -n lerobot-roco python=3.12
conda activate lerobot-roco
python -m pip install --upgrade pip
python -m pip install -e "integrations/lerobot_roco/client[lerobot,test]"
python -m pip install "PyYAML==6.0.1"
```

If you already have a working LeRobot environment and prefer to run the repo
scripts directly from source, make sure that environment has at least:
- `lerobot`
- `gymnasium`
- `pyzmq`
- `msgpack`

The repo scripts already handle LeRobot import-path differences across released
versions, so you do not need to normalize that manually.

## Phase Guide
Use the classic RoCo setup above if you only want the original LLM-driven
dialog/planning runs. For the full bridge, dataset, training, and learned-skill
pipeline, use these runbooks:

- Phase 0 bridge setup and smoke demos: [docs/phase0_setup_install_test_demos.md](docs/phase0_setup_install_test_demos.md)
- Phase 1 skill planning and execution: [docs/phase1_setup_install_test_demos.md](docs/phase1_setup_install_test_demos.md)
- Phase 2 expert dataset pipeline: [docs/phase2_setup_install_test_demos.md](docs/phase2_setup_install_test_demos.md)
- Phase 3 ACT training: [docs/phase3_short_tutorial.md](docs/phase3_short_tutorial.md)
- Phase 4 direct ACT rollout through the bridge: [docs/phase4_direct_inference.md](docs/phase4_direct_inference.md)
- Phase 5 learned executor and deployment boundary: [docs/phase5_short_tutorial.md](docs/phase5_short_tutorial.md)
- Phase 8 benchmark release and validation: [docs/phase8_benchmark_release.md](docs/phase8_benchmark_release.md)
- Phase 8 full benchmark demo: [docs/phase8_demo_tutorial.md](docs/phase8_demo_tutorial.md)

### Verified debug path
The following phase commands were re-run during this verification pass:

- Phase 1: `python scripts/smoke_test_pack_skill.py` and `python scripts/smoke_test_pack_skill.py --execute`
- Phase 0: `scripts/start_roco_bridge.py`, `scripts/smoke_test_roco_bridge.py`, and `scripts/smoke_test_lerobot_preprocessing.py`
- Phase 2: `scripts/collect_roco_expert_dataset.py`, `scripts/validate_roco_dataset.py`, `scripts/create_roco_dataset_splits.py`, `scripts/visualize_roco_dataset.py`, `scripts/replay_roco_dataset_episode.py`, and `export_local_dataset_to_lerobot(...)`
- Phase 3: `scripts/train_roco_act.py --dry-run`, full debug ACT training, and `scripts/inspect_roco_checkpoint.py`
- Phase 4: `scripts/rollout_roco_policy.py` and `scripts/inspect_roco_rollout.py`
- Phase 5: learned-executor tests plus the Phase 1/4 runtime checks above

The current debug ACT checkpoint is useful for pipeline validation, but on the
verified environment it terminated immediately with `ACTION_OUT_OF_BOUNDS`, so
do not treat the Phase 4 debug rollout as manipulation success.

### Classic usage
Run the original PackGrocery dialog loop from the Python 3.8 `roco`
environment:
```
python run_dialog.py --task pack -llm gpt-4
```

### Gemini Flash-Lite on Vertex AI
The repo's default conda environment in this README uses Python 3.8, but Google's `google-genai` SDK requires Python 3.9+.

If you want to use Gemini on Vertex AI while keeping the rest of the repo on Python 3.8, create a small helper environment:
```
python3.10 -m venv .venv-google-genai
.venv-google-genai/bin/pip install --upgrade pip
.venv-google-genai/bin/pip install google-genai
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

If you run the repo from Python 3.9+, you can just install `google-genai` in that environment directly.

### Optional visualization dependency
`open3d` is only required for voxel scene visualization and point cloud helpers. Headless runs such as `run_dialog.py --skip_display` no longer require it at import time.

Install it only if you want those visualization features:
```
pip install open3d
# or
pip install -e ".[visualization]"
```

### Acquire OpenAI or Gemini credentials
See [`docs/llm_setup_and_credentials.md`](docs/llm_setup_and_credentials.md)
for the exact lookup order and runner-specific behavior. In short, the legacy
`run_dialog.py` path still reads `./openai_key.json`, while newer paths that
call `llm_api.py` can use OpenAI environment variables or Gemini/Vertex ADC.

Gemini models now use Vertex AI with Application Default Credentials (ADC). For a local service account JSON flow:

1. Save the JSON key locally
```
mkdir -p ~/.secrets
nano ~/.secrets/bloom-gemini-sa.json
chmod 600 ~/.secrets/bloom-gemini-sa.json
```

2. Export the ADC and Vertex project settings
```
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
```

3. Optional: if the main repo is still running on Python 3.8, point Gemini calls at the helper interpreter
```
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

You can also pass `--api_key_path /path/to/service-account.json` to runners
that call `llm_api.py`; for Gemini models this sets
`GOOGLE_APPLICATION_CREDENTIALS` for the current process.

### Run MetaWorld with Gemini Flash-Lite
```
$ conda activate roco
(roco) $ export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
(roco) $ export GOOGLE_CLOUD_PROJECT="bloom-475216"
(roco) $ export GOOGLE_CLOUD_LOCATION="global"
(roco) $ export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
(roco) $ python run_metaworld_dialog.py --task pick-place-v3 --control_mode llm --llm_source gemini-2.5-flash-lite
```

### Or point at a specific service account JSON file
```
$ python run_metaworld_dialog.py --task pick-place-v3 --control_mode llm --llm_source gemini-2.5-flash-lite --api_key_path ~/.secrets/bloom-gemini-sa.json
```

### Standalone Flash-Lite smoke test
```
$ export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
$ export GOOGLE_CLOUD_PROJECT="bloom-475216"
$ export GOOGLE_CLOUD_LOCATION="global"
$ .venv-google-genai/bin/python test_gemini_lite.py
```

## MetaWorld Integration
This branch also adds a lightweight bridge for running prompt-based experiments on [MetaWorld](https://github.com/Farama-Foundation/Metaworld) while keeping the original repo usable on Python 3.8.

MetaWorld 3.x requires Python 3.10+, so the integration runs it in a helper interpreter:
```
python3.10 -m venv .venv-metaworld
.venv-metaworld/bin/pip install --upgrade pip
.venv-metaworld/bin/pip install "gymnasium==1.1.0" "mujoco==3.3.0" "metaworld==3.0.0" imageio
export METAWORLD_PYTHON_BIN="$PWD/.venv-metaworld/bin/python"
```

If you prefer to point at a local clone instead of an installed package:
```
export METAWORLD_REPO_DIR=/path/to/Farama-Foundation/Metaworld
```

The helper auto-patches MetaWorld's legacy render metadata so recent Gymnasium
MuJoCo builds do not fail on the newer `rgbd_tuple` render-mode assertion.

### Smoke-test the bridge with MetaWorld's scripted expert
```
python run_metaworld_dialog.py --task reach-v3 --control_mode expert --save_images
```

### Run the same loop with an LLM controller
```
python run_metaworld_dialog.py --task pick-place-v3 --control_mode llm --llm_source gpt-4
```

The runner writes step logs under `data/<run_name>/...` and also emits a `lerobot_steps.jsonl` file with a LeRobot-friendly schema:
- `observation.state`
- `observation.image_path`
- `action`
- `task`, `task_id`, `task_one_hot`
- reward and success flags

That export is intentionally lightweight, so it is easy to convert into a full LeRobot dataset pipeline later without coupling this repo directly to LeRobot internals.


## Contact
Please direct to [Mandi Zhao](https://mandizhao.github.io). 
If you are interested in contributing or collaborating, please feel free to reach out! I'm more than happy to chat and brainstorm together. 

## Cite
```
@misc{mandi2023roco,
      title={RoCo: Dialectic Multi-Robot Collaboration with Large Language Models}, 
      author={Zhao Mandi and Shreeya Jain and Shuran Song},
      year={2023},
      eprint={2307.04738},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
