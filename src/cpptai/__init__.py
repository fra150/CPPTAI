"""CPPTAI package initializer.

Exports the main orchestrator and core types for convenience.
"""
from .types import (
    BenchmarkItem,
    DifficultyLevel,
    PhaseOutput,
    ProblemBlock,
    RunArtifact,
    RunConfig,
    SolutionState,
)
from .core import (
    EntropicSegregator,
    VerticalTopology,
    DescentVector,
    ConvergenceProtocol,
    ComplexityScorer,
    SemanticGradient,
    ConsistencyEnforcer,
    CPPTAITraslocatore,
)
from .pipeline_v2 import run as run_v2

__all__ = [
    "BenchmarkItem",
    "DifficultyLevel",
    "PhaseOutput",
    "ProblemBlock",
    "RunArtifact",
    "RunConfig",
    "SolutionState",
    "EntropicSegregator",
    "VerticalTopology",
    "DescentVector",
    "ConvergenceProtocol",
    "ComplexityScorer",
    "SemanticGradient",
    "ConsistencyEnforcer",
    "CPPTAITraslocatore",
    "run_v2",
]

