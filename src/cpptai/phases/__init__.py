"""Phase modules for CPPTAI v2 execution."""

from .phase1 import run as run_phase1
from .phase2 import run as run_phase2
from .phase3 import run as run_phase3
from .phase4 import run as run_phase4
from .phase5 import run as run_phase5

__all__ = [
    "run_phase1",
    "run_phase2",
    "run_phase3",
    "run_phase4",
    "run_phase5",
]
