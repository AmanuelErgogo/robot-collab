import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import openai


OPENAI_API_BASE = "https://api.openai.com/v1"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GEMINI_LOCATION = "global"


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    auth_mode: str
    key_env_vars: Tuple[str, ...] = ()
    key_path_env_vars: Tuple[str, ...] = ()
    default_key_filenames: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Dict[str, object]


class BaseLLMClient:
    def __init__(self, llm_source: str, provider_spec: ProviderSpec):
        self.llm_source = llm_source
        self.provider_spec = provider_spec

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        response = openai.ChatCompletion.create(
            model=self.llm_source,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            text=response["choices"][0]["message"]["content"],
            usage=dict(response["usage"]),
        )


class VertexGeminiClient(BaseLLMClient):
    def __init__(self, llm_source: str, provider_spec: ProviderSpec, api_key_path: Optional[str] = None):
        super().__init__(llm_source, provider_spec)
        self.credentials_path = configure_vertex_adc(api_key_path=api_key_path)
        self.project = resolve_google_cloud_project(self.credentials_path)
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_GEMINI_LOCATION)
        self._direct_client = None
        if _can_use_direct_google_genai():
            self._direct_client = _create_direct_google_genai_client(
                project=self.project,
                location=self.location,
            )
        else:
            self.helper_python = resolve_google_genai_python_bin()

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        system_instruction, contents = split_messages_for_gemini(messages)
        if self._direct_client is not None:
            return _generate_with_direct_google_genai(
                client=self._direct_client,
                model=self.llm_source,
                system_instruction=system_instruction,
                contents=contents,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return _generate_with_google_genai_bridge(
            python_bin=self.helper_python,
            model=self.llm_source,
            project=self.project,
            location=self.location,
            system_instruction=system_instruction,
            contents=contents,
            max_tokens=max_tokens,
            temperature=temperature,
        )


PROVIDER_SPECS = {
    "openai": ProviderSpec(
        provider="openai",
        auth_mode="api_key",
        key_env_vars=("OPENAI_API_KEY", "LLM_API_KEY"),
        key_path_env_vars=("OPENAI_API_KEY_PATH", "LLM_API_KEY_PATH"),
        default_key_filenames=("openai_key.json",),
    ),
    "gemini": ProviderSpec(
        provider="gemini",
        auth_mode="vertex_adc",
        key_path_env_vars=("GOOGLE_APPLICATION_CREDENTIALS", "GEMINI_API_KEY_PATH", "LLM_API_KEY_PATH"),
        default_key_filenames=("gemini_key.json", "gemini_api_key.json", "google_api_key.json"),
    ),
}


def detect_provider(llm_source: str) -> str:
    normalized = llm_source.strip().lower()
    if len(normalized) == 0:
        raise ValueError("llm_source must be a non-empty model name.")
    if normalized.startswith("gemini"):
        return "gemini"
    if normalized.startswith("claude"):
        raise NotImplementedError(
            "Claude model names are not wired into this repo. "
            "Use an OpenAI model like 'gpt-4' or a Gemini model like 'gemini-2.5-flash-lite'."
        )
    return "openai"


def create_llm_client(llm_source: str, api_key_path: Optional[str] = None) -> BaseLLMClient:
    provider = detect_provider(llm_source)
    spec = PROVIDER_SPECS[provider]
    if provider == "openai":
        configure_openai_client(llm_source=llm_source, api_key_path=api_key_path)
        return OpenAIClient(llm_source=llm_source, provider_spec=spec)
    return VertexGeminiClient(llm_source=llm_source, provider_spec=spec, api_key_path=api_key_path)


def configure_openai_client(llm_source: str, api_key_path: Optional[str] = None) -> ProviderSpec:
    provider = detect_provider(llm_source)
    if provider != "openai":
        raise ValueError("configure_openai_client only supports OpenAI models.")
    spec = PROVIDER_SPECS[provider]
    api_key = resolve_openai_api_key(spec, llm_source=llm_source, api_key_path=api_key_path)
    openai.api_key = api_key
    openai.api_base = OPENAI_API_BASE
    openai.api_type = "open_ai"
    openai.api_version = None
    openai.organization = None
    return spec


def configure_vertex_adc(api_key_path: Optional[str] = None) -> str:
    if api_key_path is not None:
        explicit_path = resolve_existing_path(api_key_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(explicit_path)
        return str(explicit_path)

    configured_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if configured_path:
        existing = resolve_existing_path(configured_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(existing)
        return str(existing)

    spec = PROVIDER_SPECS["gemini"]
    for filename in spec.default_key_filenames:
        candidate = REPO_ROOT / filename
        if candidate.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
            return str(candidate)

    raise FileNotFoundError(
        "Gemini Vertex AI access requires Application Default Credentials. "
        "Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file or pass --api_key_path."
    )


def resolve_openai_api_key(
    spec: ProviderSpec,
    llm_source: str,
    api_key_path: Optional[str] = None,
) -> str:
    if api_key_path is not None:
        explicit_path = resolve_existing_path(api_key_path)
        return _read_api_key_file(explicit_path)

    api_key = _read_env_api_key(spec.key_env_vars)
    if api_key is not None:
        return api_key

    candidate_paths = []
    for env_var in spec.key_path_env_vars:
        configured_path = os.environ.get(env_var)
        if configured_path:
            candidate_paths.extend(_expand_path_candidates(configured_path))

    for filename in spec.default_key_filenames:
        candidate_paths.append(REPO_ROOT / filename)

    for candidate in _unique_paths(candidate_paths):
        if candidate.exists():
            return _read_api_key_file(candidate)

    searched = ", ".join(
        ["openai env vars: " + "/".join(spec.key_env_vars)]
        + ["openai key files: " + ", ".join(spec.default_key_filenames)]
    )
    raise FileNotFoundError(
        "Unable to locate an API key for provider '{}' while configuring model '{}'. "
        "Set one of {} or pass --api_key_path.".format(
            spec.provider,
            llm_source,
            searched,
        )
    )


def resolve_google_cloud_project(credentials_path: str) -> str:
    configured_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if configured_project and len(configured_project.strip()) > 0:
        return configured_project.strip()

    payload = json.loads(Path(credentials_path).read_text())
    project_id = payload.get("project_id")
    if isinstance(project_id, str) and len(project_id.strip()) > 0:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id.strip()
        return project_id.strip()

    raise ValueError(
        "GOOGLE_CLOUD_PROJECT is not set, and the service account file does not contain project_id."
    )


def resolve_google_genai_python_bin() -> str:
    candidates = (
        os.environ.get("GOOGLE_GENAI_PYTHON_BIN"),
        str(REPO_ROOT / ".venv-google-genai" / "bin" / "python"),
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.10"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path)
    raise RuntimeError(
        "Gemini access requires google-genai on Python 3.9+. "
        "Set GOOGLE_GENAI_PYTHON_BIN to a Python 3.9+ interpreter with google-genai installed."
    )


def split_messages_for_gemini(messages: Sequence[Dict[str, str]]) -> Tuple[Optional[str], str]:
    system_parts: List[str] = []
    content_parts: List[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_parts.append(content)
        else:
            content_parts.append(content if role == "user" else f"[{role.upper()}]\n{content}")

    system_instruction = "\n\n".join(system_parts).strip() or None
    contents = "\n\n".join(content_parts).strip()
    if len(contents) == 0:
        contents = "Respond to the instructions above."
    return system_instruction, contents


def _can_use_direct_google_genai() -> bool:
    if sys.version_info < (3, 9):
        return False
    try:
        from google import genai  # noqa: F401
        from google.genai import types  # noqa: F401
    except ImportError:
        return False
    return True


def _create_direct_google_genai_client(project: str, location: str):
    from google import genai

    return genai.Client(vertexai=True, project=project, location=location)


def _generate_with_direct_google_genai(
    client,
    model: str,
    system_instruction: Optional[str],
    contents: str,
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return LLMResponse(
        text=response.text,
        usage=normalize_gemini_usage(response.usage_metadata),
    )


def _generate_with_google_genai_bridge(
    python_bin: str,
    model: str,
    project: str,
    location: str,
    system_instruction: Optional[str],
    contents: str,
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    payload = {
        "model": model,
        "project": project,
        "location": location,
        "system_instruction": system_instruction,
        "contents": contents,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    completed = subprocess.run(
        [python_bin, str(REPO_ROOT / "vertex_gemini_bridge.py")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Gemini bridge call failed with code {}.\nstdout:\n{}\nstderr:\n{}".format(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
        )
    response_payload = json.loads(completed.stdout)
    return LLMResponse(
        text=response_payload["text"],
        usage=response_payload["usage"],
    )


def normalize_gemini_usage(usage_metadata) -> Dict[str, object]:
    if usage_metadata is None:
        return {}
    return {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
        "total_tokens": getattr(usage_metadata, "total_token_count", None),
        "traffic_type": str(getattr(usage_metadata, "traffic_type", "")),
    }


def resolve_existing_path(path_str: str) -> Path:
    explicit_candidates = _unique_paths(_expand_path_candidates(path_str))
    for candidate in explicit_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Configured path '{}' could not be found.".format(path_str))


def _read_env_api_key(env_var_names: Iterable[str]) -> Optional[str]:
    for env_var in env_var_names:
        api_key = os.environ.get(env_var)
        if api_key is not None and len(api_key.strip()) > 0:
            return api_key.strip()
    return None


def _expand_path_candidates(path_str: str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return [path]
    return [Path.cwd() / path, REPO_ROOT / path]


def _unique_paths(paths: Iterable[Path]):
    seen = set()
    unique = []
    for path in paths:
        resolved = str(path.resolve(strict=False))
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _read_api_key_file(path: Path) -> str:
    raw_text = path.read_text().strip()
    if len(raw_text) == 0:
        raise ValueError("API key file '{}' is empty.".format(path))

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = raw_text
    return _normalize_api_key_payload(payload, path)


def _normalize_api_key_payload(payload, path: Path) -> str:
    if isinstance(payload, str):
        api_key = payload.strip()
        if len(api_key) == 0:
            raise ValueError("API key file '{}' does not contain a usable key.".format(path))
        return api_key

    if isinstance(payload, dict):
        for key_name in ("api_key", "key", "token", "value"):
            api_key = payload.get(key_name)
            if isinstance(api_key, str) and len(api_key.strip()) > 0:
                return api_key.strip()

    raise ValueError(
        "API key file '{}' must contain either a JSON string, raw key text, or a JSON object "
        "with one of: api_key, key, token, value.".format(path)
    )
