import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from rocobench.skills.models import SkillCall, SkillPlan
from rocobench.skills.registry import SkillRegistry


LOGGER = logging.getLogger(__name__)
_AGENT_LINE_RE = re.compile(r"^NAME\s+(\S+)\s+ACTION\s+(.+)$")
_SKILL_CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\- ]*)\((.*)\)$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNQUOTED_VALUE_RE = re.compile(r"^[A-Za-z0-9_\-./]+$")


class SkillResponseParser:
    """Parse strict coordinated skill-call responses from an LLM."""

    def __init__(self, registry: SkillRegistry, agent_names: Sequence[str]) -> None:
        self.registry = registry
        self.agent_names = list(agent_names)

    def parse(self, obs, response: str) -> Tuple[bool, str, List[SkillPlan]]:
        try:
            return self._parse(obs, response)
        except Exception:
            LOGGER.exception("Unexpected skill parser failure")
            return False, "Internal parser error. Re-format the EXECUTE block and try again.", []

    def _parse(self, obs, response: str) -> Tuple[bool, str, List[SkillPlan]]:
        del obs
        if response is None or "EXECUTE" not in response:
            return False, "Response must contain an EXECUTE block.", []

        execute_body = response.split("EXECUTE", 1)[1]
        success, message, executable_lines = self._collect_agent_lines(execute_body)
        if not success:
            return False, message, []
        if not executable_lines:
            return False, "EXECUTE block must contain one NAME ... ACTION ... line per agent.", []

        calls_by_agent = {}  # type: Dict[str, SkillCall]
        parsed_lines = []
        for line in executable_lines:
            match = _AGENT_LINE_RE.match(line)
            if not match:
                return False, "Malformed action line: {}.".format(line), []
            agent_name = match.group(1).strip()
            raw_action = match.group(2).strip()
            if agent_name not in self.agent_names:
                return False, "Unknown agent {}.".format(agent_name), []
            if agent_name in calls_by_agent:
                return False, "Duplicate ACTION line for {}.".format(agent_name), []

            success, message, call = self._parse_skill_call(agent_name, raw_action)
            if not success:
                return False, message, []
            calls_by_agent[agent_name] = call
            parsed_lines.append("NAME {} ACTION {}".format(agent_name, str(call)))

        missing = [agent_name for agent_name in self.agent_names if agent_name not in calls_by_agent]
        if missing:
            return False, "Missing ACTION line for {}.".format(", ".join(missing)), []
        if len(calls_by_agent) != len(self.agent_names):
            return False, "Response must contain exactly one ACTION for each configured agent.", []

        ordered_calls = [calls_by_agent[agent_name] for agent_name in self.agent_names]
        parsed_proposal = "EXECUTE\n" + "\n".join(parsed_lines)
        plan = SkillPlan(calls=ordered_calls, parsed_proposal=parsed_proposal)
        return True, parsed_proposal, [plan]

    def _collect_agent_lines(self, execute_body: str) -> Tuple[bool, str, List[str]]:
        lines = [line.strip() for line in execute_body.splitlines()]
        collected = []
        current = None  # type: Optional[str]
        paren_balance = 0
        in_quote = None  # type: Optional[str]

        for line in lines:
            if not line:
                continue
            starts_agent = line.startswith("NAME ")
            if starts_agent and current is not None and paren_balance == 0:
                collected.append(current.strip())
                current = line
            elif starts_agent and current is None:
                current = line
            elif current is not None:
                current += " " + line
            else:
                return False, "Only NAME ... ACTION ... lines are allowed after EXECUTE.", []

            paren_balance, in_quote = self._paren_balance(current)
            if paren_balance < 0:
                return False, "Malformed parentheses in action line: {}.".format(current), []
            if current is not None and paren_balance == 0 and in_quote is None:
                collected.append(current.strip())
                current = None

        if current is not None:
            paren_balance, in_quote = self._paren_balance(current)
            if paren_balance != 0 or in_quote is not None:
                return False, "Malformed parentheses in action line: {}.".format(current), []
            collected.append(current.strip())

        return True, "", collected

    def _paren_balance(self, text: str) -> Tuple[int, Optional[str]]:
        balance = 0
        in_quote = None  # type: Optional[str]
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if in_quote:
                if char == in_quote:
                    in_quote = None
                continue
            if char in ("'", '"'):
                in_quote = char
                continue
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
        return balance, in_quote

    def _parse_skill_call(self, agent_name: str, raw_action: str) -> Tuple[bool, str, Optional[SkillCall]]:
        match = _SKILL_CALL_RE.match(raw_action)
        if not match:
            return False, "Action must be a function call like WAIT() or PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left).", None
        skill_name = match.group(1).strip()
        if not self.registry.has(skill_name):
            return False, "Unknown skill {}.".format(skill_name), None
        spec = self.registry.get(skill_name)

        arg_text = match.group(2).strip()
        success, message, arguments = self._parse_arguments(arg_text)
        if not success:
            return False, message, None
        call = SkillCall(
            agent_name=agent_name,
            skill_name=spec.name,
            arguments=arguments,
            raw_action=raw_action,
        )
        call = self.registry.canonicalize_call(call)
        shape = self.registry.validate_call_shape(call)
        if not shape.valid:
            return False, shape.to_feedback(), None
        return True, "", call

    def _parse_arguments(self, arg_text: str) -> Tuple[bool, str, Dict[str, str]]:
        if not arg_text:
            return True, "", {}
        parts = self._split_arguments(arg_text)
        if parts is None:
            return False, "Malformed quoted argument value.", {}

        arguments = {}
        for part in parts:
            if not part:
                return False, "Empty argument is not allowed.", {}
            if "=" not in part:
                return False, "Positional arguments are not allowed; use key=value.", {}
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not _IDENT_RE.match(key):
                return False, "Invalid argument name {}.".format(key), {}
            if key in arguments:
                return False, "Duplicate argument {}.".format(key), {}
            success, message, parsed_value = self._parse_value(value)
            if not success:
                return False, message, {}
            arguments[key] = parsed_value
        return True, "", arguments

    def _split_arguments(self, arg_text: str) -> Optional[List[str]]:
        parts = []
        current = []
        in_quote = None
        escaped = False
        for char in arg_text:
            if escaped:
                current.append(char)
                escaped = False
                continue
            if char == "\\":
                current.append(char)
                escaped = True
                continue
            if in_quote:
                current.append(char)
                if char == in_quote:
                    in_quote = None
                continue
            if char in ("'", '"'):
                current.append(char)
                in_quote = char
                continue
            if char == ",":
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if in_quote:
            return None
        parts.append("".join(current).strip())
        return parts

    def _parse_value(self, value: str) -> Tuple[bool, str, str]:
        if not value:
            return False, "Argument values cannot be empty.", ""
        if value[0] in ("'", '"'):
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                return False, "Malformed quoted argument value.", ""
            inner = value[1:-1]
            parsed = []
            escaped = False
            for char in inner:
                if escaped:
                    if char in ('"', "'", "\\"):
                        parsed.append(char)
                    else:
                        parsed.append(char)
                    escaped = False
                elif char == "\\":
                    escaped = True
                else:
                    parsed.append(char)
            if escaped:
                return False, "Malformed quoted argument value.", ""
            return True, "", "".join(parsed).strip()
        if not _UNQUOTED_VALUE_RE.match(value):
            return False, "Unquoted argument values must be identifiers; quote values containing spaces.", ""
        return True, "", value
