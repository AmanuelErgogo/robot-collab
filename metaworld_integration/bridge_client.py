"""Client wrapper around the Python 3.10+ MetaWorld helper process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_metaworld_python_bin():
    candidates = (
        os.environ.get("METAWORLD_PYTHON_BIN"),
        str(REPO_ROOT / ".venv-metaworld" / "bin" / "python"),
        shutil.which("python3.12"),
        shutil.which("python3.10"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path)
    raise RuntimeError(
        "MetaWorld integration needs a Python 3.10+ interpreter. "
        "Set METAWORLD_PYTHON_BIN or create .venv-metaworld."
    )


class MetaWorldBridgeClient:
    """Stateful RPC client for the helper process defined in metaworld_bridge.py."""

    def __init__(
        self,
        task_name,
        python_bin=None,
        repo_dir=None,
        camera_name="corner2",
        width=480,
        height=480,
    ):
        self.task_name = task_name
        self.python_bin = python_bin or resolve_metaworld_python_bin()
        self.repo_dir = repo_dir or os.environ.get("METAWORLD_REPO_DIR")
        helper_path = REPO_ROOT / "metaworld_bridge.py"
        cmd = [
            self.python_bin,
            "-u",
            str(helper_path),
            "--task",
            task_name,
            "--camera-name",
            camera_name,
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        if self.repo_dir:
            cmd.extend(["--repo-dir", self.repo_dir])

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        self._await_ready()

    def _await_ready(self):
        if self.process.stdout is None:
            raise RuntimeError("MetaWorld helper stdout is unavailable.")
        ready_line = self.process.stdout.readline()
        if len(ready_line) == 0:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "MetaWorld helper failed to start for task '{}'. stderr:\n{}".format(
                    self.task_name,
                    stderr,
                )
            )
        payload = json.loads(ready_line)
        if not payload.get("ok", False):
            raise RuntimeError("MetaWorld helper failed to initialize: {}".format(payload.get("error")))
        self.metadata = payload.get("data", {})

    def _request(self, command, **kwargs):
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("MetaWorld helper pipes are unavailable.")
        request_payload = {"command": command, "kwargs": kwargs}
        self.process.stdin.write(json.dumps(request_payload) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if len(response_line) == 0:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "MetaWorld helper exited while handling '{}'. stderr:\n{}".format(command, stderr)
            )
        response_payload = json.loads(response_line)
        if not response_payload.get("ok", False):
            raise RuntimeError(response_payload.get("error", "Unknown MetaWorld helper error."))
        return response_payload.get("data", {})

    def reset(self, seed=None):
        return self._request("reset", seed=seed)

    def step(self, action):
        return self._request("step", action=list(action))

    def expert_action(self):
        data = self._request("expert_action")
        return data["action"]

    def save_frame(self, path):
        return self._request("save_frame", path=path)

    def close(self):
        try:
            if self.process.poll() is None:
                self._request("close")
        except Exception:
            pass
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
