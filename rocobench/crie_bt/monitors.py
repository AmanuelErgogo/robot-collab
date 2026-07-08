"""Progress monitor backends (canonical import location).

The next-stage plan renames the confusingly named ``vlm_sarm_monitor`` module:
the simulator-coded monitor belongs to CRIE-BT, not to a baseline "VLM/SARM"
method.  Import monitors from here:

    from rocobench.crie_bt.monitors import CodedSimProgressMonitor  # CRIE-BT sim
    from rocobench.crie_bt.monitors import SARMProgressMonitor       # CRIE-BT real
    from rocobench.crie_bt.monitors import NoSeparateMonitor         # baseline placeholder

The canonical implementations live in ``rocobench.crie_bt.pipeline.monitors``.
The legacy :class:`SimulatorSignalVLMSARMMonitor` is re-exported for backward
compatibility with existing controllers/tests and is deprecated.
"""

from __future__ import annotations

from .pipeline.monitors import (
    CodedSimProgressMonitor,
    NoSeparateMonitor,
    SARMProgressMonitor,
    build_monitor,
)

# Backward-compatible alias: the old simulator-signal monitor. Prefer
# CodedSimProgressMonitor for new code.
from .vlm_sarm_monitor import (
    BaseVLMSARMMonitorBackend,
    SimulatorSignalVLMSARMMonitor,
    VLMSARMMonitorDecision,
)

__all__ = [
    "BaseVLMSARMMonitorBackend",
    "CodedSimProgressMonitor",
    "NoSeparateMonitor",
    "SARMProgressMonitor",
    "SimulatorSignalVLMSARMMonitor",
    "VLMSARMMonitorDecision",
    "build_monitor",
]
