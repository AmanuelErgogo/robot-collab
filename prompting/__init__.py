"""Prompting package exports with lazy imports for heavy legacy modules."""


def __getattr__(name):
    if name == "LLMResponseParser":
        from prompting.parser import LLMResponseParser

        return LLMResponseParser
    if name == "FeedbackManager":
        from prompting.feedback import FeedbackManager

        return FeedbackManager
    if name == "DialogPrompter":
        from prompting.dialog_prompter import DialogPrompter

        return DialogPrompter
    if name == "SingleThreadPrompter":
        from prompting.plan_prompter import SingleThreadPrompter

        return SingleThreadPrompter
    if name == "save_episode_html":
        from prompting.display_utils import save_episode_html

        return save_episode_html
    if name == "SkillResponseParser":
        from prompting.skill_parser import SkillResponseParser

        return SkillResponseParser
    if name == "SkillFeedbackManager":
        from prompting.skill_feedback import SkillFeedbackManager

        return SkillFeedbackManager
    if name in ("LegacyPlanningPromptProvider", "PackGrocerySkillPromptProvider", "PlanningPromptProvider"):
        from prompting.skill_prompt import (
            LegacyPlanningPromptProvider,
            PackGrocerySkillPromptProvider,
            PlanningPromptProvider,
        )

        return {
            "LegacyPlanningPromptProvider": LegacyPlanningPromptProvider,
            "PackGrocerySkillPromptProvider": PackGrocerySkillPromptProvider,
            "PlanningPromptProvider": PlanningPromptProvider,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DialogPrompter",
    "FeedbackManager",
    "LLMResponseParser",
    "LegacyPlanningPromptProvider",
    "PackGrocerySkillPromptProvider",
    "PlanningPromptProvider",
    "SingleThreadPrompter",
    "SkillFeedbackManager",
    "SkillResponseParser",
    "save_episode_html",
]
