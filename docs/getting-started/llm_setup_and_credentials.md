# LLM Setup And Credentials

This repo has two LLM paths with different credential behavior.

## 1. Classic RoCo Dialog

`run_dialog.py` uses the legacy prompting modules:

```text
run_dialog.py
  -> prompting.plan_prompter.SingleThreadPrompter
  -> prompting.dialog_prompter.DialogPrompter
  -> openai.ChatCompletion.create(...)
```

This path currently supports the legacy OpenAI flow. It does not use
`llm_api.py`, and it does not currently pass `--api_key_path` through to the
prompting code.

Credential location:

```text
openai_key.json
```

The file is expected at the repository root:

```text
/home/amanu/robot-collab/openai_key.json
```

The legacy loader expects a JSON string:

```json
"sk-..."
```

Run example:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python run_dialog.py \
  --task pack \
  -llm gpt-4 \
  --output_mode action_only \
  --comm_mode plan \
  -t 1 \
  -nruns 1 \
  -sd
```

You can also run with `--debug_mode` to type actions manually and avoid an LLM
call.

## 2. Shared Provider Wrapper

`llm_api.py` is the newer provider wrapper. It is used by:

```text
run_metaworld_dialog.py
real_world/runners/dialog_runner.py
real_world/prompts/dialog_prompter.py
```

This path supports OpenAI models and Gemini models through Vertex AI.

OpenAI lookup order:

```text
OPENAI_API_KEY
LLM_API_KEY
OPENAI_API_KEY_PATH
LLM_API_KEY_PATH
openai_key.json
```

If a key file is used, it may contain raw key text, a JSON string, or a JSON
object with one of these fields:

```text
api_key
key
token
value
```

Gemini/Vertex lookup order:

```text
GOOGLE_APPLICATION_CREDENTIALS
GEMINI_API_KEY_PATH
LLM_API_KEY_PATH
gemini_key.json
gemini_api_key.json
google_api_key.json
```

Gemini uses Vertex AI Application Default Credentials. The file should be a
Google service account JSON file, not a simple API key string.

Recommended local Gemini setup:

```bash
mkdir -p "$HOME/.secrets"
chmod 700 "$HOME/.secrets"

export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

The `GOOGLE_GENAI_PYTHON_BIN` helper is needed when the main RoCo environment is
Python 3.8, because Google's `google-genai` SDK requires Python 3.9 or newer.

MetaWorld LLM example:

```bash
python run_metaworld_dialog.py \
  --task pick-place-v3 \
  --control_mode llm \
  --llm_source gemini-2.5-flash-lite \
  --api_key_path "$HOME/.secrets/bloom-gemini-sa.json"
```

OpenAI through the shared wrapper:

```bash
export OPENAI_API_KEY="sk-..."
python run_metaworld_dialog.py \
  --task pick-place-v3 \
  --control_mode llm \
  --llm_source gpt-4
```

## Local Secret Storage

Do not commit credential files. The repository `.gitignore` excludes:

```text
openai_key.json
claude_key.json
.env
.secrets/
```

Safe places to store local credentials:

```text
$HOME/.secrets/
/home/amanu/robot-collab/.secrets/
```

Use restrictive permissions:

```bash
chmod 600 /path/to/key.json
```

Do not print key contents in logs, command output, docs, or test artifacts.

## Quick Checks

Check whether expected environment variables are set without printing values:

```bash
python - <<'PY'
import os

for name in [
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "OPENAI_API_KEY_PATH",
    "LLM_API_KEY_PATH",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GEMINI_API_KEY_PATH",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_PYTHON_BIN",
]:
    print("{}: {}".format(name, "set" if os.environ.get(name) else "unset"))
PY
```

Check whether root-level key files exist without reading them:

```bash
test -f openai_key.json && echo "openai_key.json present" || echo "openai_key.json absent"
test -f gemini_key.json && echo "gemini_key.json present" || echo "gemini_key.json absent"
```

## Current Limitations

- `run_dialog.py` is still wired to the legacy OpenAI prompting path.
- Gemini examples should use code paths that call `llm_api.py`, such as
  `run_metaworld_dialog.py`, unless `run_dialog.py` is later refactored.
- Claude model names are rejected by `llm_api.py`; use OpenAI or Gemini.
- The `prompting` package imports `openai` at module load, so `openai` must be
  importable in the run env **even when using Gemini** (it is not called on the
  Gemini path). Install once if missing: `pip install "openai==0.28.1"`.
