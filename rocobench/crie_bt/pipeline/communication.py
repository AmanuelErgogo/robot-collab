"""Human/robot communication interfaces.

Step 2 begins with a terminal/keyboard human interface
(:class:`TerminalCommunicationInterface`).  Speech (TTS/STT) is added later as a
wrapper around the *same* send/read API, so controller logic never changes.
:class:`ScriptedHumanCommunicationInterface` replays a fixed response queue for
automated runs and tests.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .interfaces import CommunicationInterface, HumanInstruction, HumanResponse

# Terminal command shortcuts (see docs/crie_next_stage_plan/05_implementation_plan.md,
# Milestone 2). Numbers map to the primary responses; ``done`` marks completion.
_ACTION_BY_TOKEN = {
    "1": "accept",
    "accept": "accept",
    "yes": "accept",
    "y": "accept",
    "ok": "accept",
    "2": "reject",
    "reject": "reject",
    "no": "reject",
    "n": "reject",
    "3": "counter_propose",
    "counter": "counter_propose",
    "counter_propose": "counter_propose",
    "propose": "counter_propose",
    "done": "done",
    "complete": "done",
    "finished": "done",
}


def parse_human_response(text: str) -> HumanResponse:
    """Map a raw terminal string to a typed :class:`HumanResponse`."""
    raw = (text or "").strip()
    token = raw.lower()
    action = _ACTION_BY_TOKEN.get(token)
    if action is None:
        # First word may be a command followed by free-text (e.g. "counter use bottle B").
        first = token.split(maxsplit=1)[0] if token else ""
        action = _ACTION_BY_TOKEN.get(first, "unknown")
    return HumanResponse(text=raw, action=action, timestamp=time.time())


class TerminalCommunicationInterface(CommunicationInterface):
    """Printed messages + ``input()`` for a human collaborator at the keyboard."""

    def __init__(self, prompt: str = "human> ") -> None:
        self.prompt = prompt
        self.transcript: List[dict] = []

    def send_to_human(self, instruction: HumanInstruction) -> None:
        self.transcript.append({"role": "robot_to_human", "text": instruction.text})
        print("\n[ROBOT -> HUMAN] {}".format(instruction.text))
        print("  Respond: 1=accept  2=reject  3=counter-propose  done=completed "
              "(free text allowed).")

    def read_human_response(self, timeout_s: Optional[float] = None) -> HumanResponse:
        try:
            raw = input(self.prompt)
        except EOFError:
            raw = "done"
        response = parse_human_response(raw)
        self.transcript.append({"role": "human_to_robot", "text": response.text, "action": response.action})
        return response

    def send_robot_dialogue(self, speaker_id: str, text: str) -> None:
        self.transcript.append({"role": "robot_dialogue", "speaker": speaker_id, "text": text})
        print("[{}] {}".format(speaker_id, text))


class ScriptedHumanCommunicationInterface(CommunicationInterface):
    """Replays a fixed queue of human responses; used for automated runs/tests.

    ``responses`` may be raw strings (parsed via :func:`parse_human_response`) or
    ready-made :class:`HumanResponse` objects.  Once exhausted, it returns
    ``default_action`` so an episode never blocks.
    """

    def __init__(self, responses=None, default_action: str = "done") -> None:
        self._queue: List[HumanResponse] = []
        for item in responses or []:
            if isinstance(item, HumanResponse):
                self._queue.append(item)
            else:
                self._queue.append(parse_human_response(str(item)))
        self.default_action = default_action
        self.transcript: List[dict] = []
        self.instructions: List[HumanInstruction] = []

    def send_to_human(self, instruction: HumanInstruction) -> None:
        self.instructions.append(instruction)
        self.transcript.append({"role": "robot_to_human", "text": instruction.text})

    def read_human_response(self, timeout_s: Optional[float] = None) -> HumanResponse:
        if self._queue:
            response = self._queue.pop(0)
        else:
            response = HumanResponse(text=self.default_action, action=self.default_action, timestamp=time.time())
        self.transcript.append({"role": "human_to_robot", "text": response.text, "action": response.action})
        return response

    def send_robot_dialogue(self, speaker_id: str, text: str) -> None:
        self.transcript.append({"role": "robot_dialogue", "speaker": speaker_id, "text": text})


__all__ = [
    "ScriptedHumanCommunicationInterface",
    "TerminalCommunicationInterface",
    "parse_human_response",
]
