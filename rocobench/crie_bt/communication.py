"""Research communication abstraction for CRIE-BT runtime events."""

from typing import Dict, List

from .types import RuntimeEvent


class CommunicationManager(object):
    def handle_event(self, event: RuntimeEvent) -> List[str]:
        utterance = self.generate_utterance(event)
        return [utterance] if utterance else []

    def generate_utterance(self, event: RuntimeEvent) -> str:
        event_type = event.event_type
        skill = event.skill_call
        obj = skill.arguments.get("object", "the object") if skill is not None else "the object"
        target = ""
        if skill is not None:
            target = skill.arguments.get("container", skill.arguments.get("target", "the target"))
        if event_type == "SUBTASK_STARTED":
            return "I am working on {}.".format(skill.instruction if skill and skill.instruction else obj)
        if event_type == "SUBTASK_COMPLETED":
            return "I completed the current subtask."
        if event_type == "TASK_COMPLETE":
            return "The task is complete."
        if event_type == "LOW_CONFIDENCE" or event_type == "EXPLAIN":
            return "I am not fully confident, but progress is continuing."
        if event_type == "LOCAL_RETRY":
            message = event.message.lower()
            if "slip" in message:
                return "The object slipped. I will retry locally."
            if "grasp" in message:
                return "I missed the grasp and will retry."
            return "I will retry the current subtask."
        if event_type == "REQUEST_HUMAN_INPUT":
            return "Can you help confirm the target?"
        if event_type == "REQUEST_REPLAN":
            return "I need to replan because {}.".format(event.message or "the current subtask failed")
        if event_type == "EXECUTOR_FEEDBACK" and target:
            return ""
        return ""

    def generate_overlay(self, event: RuntimeEvent) -> Dict[str, object]:
        skill = event.skill_call
        decision = event.decision.decision.value if event.decision is not None else event.event_type
        args = dict(skill.arguments) if skill is not None else {}
        return {
            "agent": skill.agent if skill is not None else None,
            "subtask": skill.skill_name if skill is not None else None,
            "object": args.get("object"),
            "target": args.get("container", args.get("target")),
            "status": decision,
            "confidence": event.payload.get("confidence"),
            "message": event.message,
        }
