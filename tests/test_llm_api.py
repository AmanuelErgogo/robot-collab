import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import openai

from llm_api import (
    DEFAULT_GEMINI_LOCATION,
    OPENAI_API_BASE,
    VertexGeminiClient,
    configure_openai_client,
    create_llm_client,
    detect_provider,
    normalize_gemini_usage,
    resolve_google_cloud_project,
    split_messages_for_gemini,
)


class LLMApiTests(unittest.TestCase):
    def setUp(self):
        self._original_state = {
            "api_key": openai.api_key,
            "api_base": openai.api_base,
            "api_type": openai.api_type,
            "api_version": openai.api_version,
            "organization": openai.organization,
        }

    def tearDown(self):
        openai.api_key = self._original_state["api_key"]
        openai.api_base = self._original_state["api_base"]
        openai.api_type = self._original_state["api_type"]
        openai.api_version = self._original_state["api_version"]
        openai.organization = self._original_state["organization"]

    def test_detect_provider(self):
        self.assertEqual(detect_provider("gemini-2.5-flash-lite"), "gemini")
        self.assertEqual(detect_provider("gpt-4.1"), "openai")

    def test_claude_model_names_raise_helpful_error(self):
        with self.assertRaises(NotImplementedError):
            detect_provider("claude-3-7-sonnet")

    def test_configures_openai_from_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-env-key"}, clear=True):
            spec = configure_openai_client("gpt-4")

        self.assertEqual(spec.provider, "openai")
        self.assertEqual(openai.api_key, "openai-env-key")
        self.assertEqual(openai.api_base, OPENAI_API_BASE)
        self.assertEqual(openai.api_type, "open_ai")

    def test_resolves_google_cloud_project_from_service_account_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cred_path = os.path.join(temp_dir, "sa.json")
            with open(cred_path, "w") as handle:
                json.dump({"project_id": "bloom-475216"}, handle)

            with patch.dict(os.environ, {}, clear=True):
                project = resolve_google_cloud_project(cred_path)
                self.assertEqual(os.environ["GOOGLE_CLOUD_PROJECT"], "bloom-475216")

        self.assertEqual(project, "bloom-475216")

    def test_split_messages_for_gemini(self):
        system_instruction, contents = split_messages_for_gemini(
            [
                {"role": "system", "content": "Be careful."},
                {"role": "user", "content": "Plan the next step."},
            ]
        )
        self.assertEqual(system_instruction, "Be careful.")
        self.assertEqual(contents, "Plan the next step.")

    def test_gemini_client_uses_bridge_when_direct_sdk_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cred_path = os.path.join(temp_dir, "sa.json")
            with open(cred_path, "w") as handle:
                json.dump({"project_id": "bloom-475216"}, handle)

            completed = MagicMock()
            completed.returncode = 0
            completed.stdout = json.dumps(
                {
                    "text": "EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                }
            )
            completed.stderr = ""

            with patch.dict(os.environ, {}, clear=True):
                with patch("llm_api._can_use_direct_google_genai", return_value=False):
                    with patch("llm_api.resolve_google_genai_python_bin", return_value="/tmp/python3.10"):
                        with patch("llm_api.subprocess.run", return_value=completed) as run_mock:
                            client = create_llm_client(
                                "gemini-2.5-flash-lite",
                                api_key_path=cred_path,
                            )
                            response = client.generate(
                                messages=[
                                    {"role": "system", "content": "You are a robot planner."},
                                    {"role": "user", "content": "Respond with EXECUTE."},
                                ],
                                max_tokens=64,
                                temperature=0.0,
                            )
                            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], cred_path)

        self.assertIsInstance(client, VertexGeminiClient)
        self.assertEqual(client.project, "bloom-475216")
        self.assertEqual(client.location, DEFAULT_GEMINI_LOCATION)
        self.assertIn("EXECUTE", response.text)
        self.assertEqual(response.usage["total_tokens"], 14)
        run_mock.assert_called_once()

    def test_normalize_gemini_usage(self):
        usage = MagicMock()
        usage.prompt_token_count = 7
        usage.candidates_token_count = 3
        usage.total_token_count = 10
        usage.traffic_type = "ON_DEMAND"
        normalized = normalize_gemini_usage(usage)
        self.assertEqual(
            normalized,
            {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "traffic_type": "ON_DEMAND",
            },
        )


if __name__ == "__main__":
    unittest.main()
