from rocobench.crie_bt.communication import CommunicationManager
from rocobench.crie_bt.status import RuntimeDecision
from rocobench.crie_bt.types import BTDecision, RuntimeEvent, SkillCall


def test_communication_manager_generates_speech_and_overlay():
    skill = SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin"})
    event = RuntimeEvent("LOCAL_RETRY", "Missed grasp; retrying.", skill_call=skill, decision=BTDecision(RuntimeDecision.LOCAL_RETRY))
    manager = CommunicationManager()
    assert "retry" in manager.generate_utterance(event).lower()
    overlay = manager.generate_overlay(event)
    assert overlay["agent"] == "Alice"
    assert overlay["object"] == "apple"
    assert overlay["status"] == "LOCAL_RETRY"
