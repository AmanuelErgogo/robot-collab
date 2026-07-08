"""CRIE-BT / SARM next-stage pipeline.

A clean, swappable architecture layer for the eight paper conditions across
Step 1 (robot-robot sim), Step 2 (human-robot sim with a terminal interface),
and Step 3 (real-world human-robot with learned skills + SARM).

Import surface is dependency-light: importing this package does not pull in
MuJoCo, dm_control, or any LLM client. Heavy backends are lazy extension points.
"""

from __future__ import annotations

from .communication import (
    ScriptedHumanCommunicationInterface,
    TerminalCommunicationInterface,
    parse_human_response,
)
from .conditions import (
    ConditionConfig,
    ConditionRegistry,
    all_condition_names,
    build_condition,
    get_registry,
)
from .controllers import (
    CRIEBTCentralizedController,
    CRIEBTDialogController,
    PipelineController,
    VLMCentralizedController,
    VLMDialogController,
    build_collaboration_controller,
    build_monitor_for_condition,
    default_planner_factory,
)
from .env_adapters import (
    SYNTHETIC_TASKS,
    StageSpec,
    SyntheticEnvironmentAdapter,
    available_synthetic_tasks,
)
from .episode_logger import EpisodeLogger, REQUIRED_ROW_FIELDS
from .executors import (
    LearnedSkillExecutorAdapter,
    RRTSkillExecutorAdapter,
    SyntheticSkillExecutor,
)
from .interfaces import (
    CollaborationController,
    CommunicationInterface,
    EnvironmentAdapter,
    ExecutionFeedback,
    HumanInstruction,
    HumanResponse,
    MonitorDecision,
    ObservationBundle,
    PlanStep,
    PlanUpdate,
    PlannerInterface,
    ProgressMonitorInterface,
    SkillCall,
    SkillExecutorInterface,
    StageStatus,
    planner_safe_observation,
)
from .monitors import (
    CodedSimProgressMonitor,
    NoSeparateMonitor,
    SARMProgressMonitor,
    build_monitor,
)
from .planners import LLMPlannerAdapter, ScriptedStagePlanner

__all__ = [
    "CRIEBTCentralizedController",
    "CRIEBTDialogController",
    "CodedSimProgressMonitor",
    "CollaborationController",
    "CommunicationInterface",
    "ConditionConfig",
    "ConditionRegistry",
    "EnvironmentAdapter",
    "EpisodeLogger",
    "ExecutionFeedback",
    "HumanInstruction",
    "HumanResponse",
    "LLMPlannerAdapter",
    "LearnedSkillExecutorAdapter",
    "MonitorDecision",
    "NoSeparateMonitor",
    "ObservationBundle",
    "PipelineController",
    "PlanStep",
    "PlanUpdate",
    "PlannerInterface",
    "ProgressMonitorInterface",
    "REQUIRED_ROW_FIELDS",
    "RRTSkillExecutorAdapter",
    "SARMProgressMonitor",
    "SYNTHETIC_TASKS",
    "ScriptedHumanCommunicationInterface",
    "ScriptedStagePlanner",
    "SkillCall",
    "SkillExecutorInterface",
    "StageSpec",
    "StageStatus",
    "SyntheticEnvironmentAdapter",
    "SyntheticSkillExecutor",
    "TerminalCommunicationInterface",
    "VLMCentralizedController",
    "VLMDialogController",
    "all_condition_names",
    "available_synthetic_tasks",
    "build_collaboration_controller",
    "build_condition",
    "build_monitor",
    "build_monitor_for_condition",
    "default_planner_factory",
    "get_registry",
    "parse_human_response",
    "planner_safe_observation",
]
