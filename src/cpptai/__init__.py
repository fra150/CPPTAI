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
    ExplorationTrajectory,
    AnalyzerPrepared,
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
from .responsible_ai import ResponsibleAIAuditor
from .pipeline_v2 import run as run_v2
from .baselines import CoTBaseline, ToTBaseline, GoTBaseline, ReActBaseline

__all__ = [
    "BenchmarkItem",
    "DifficultyLevel",
    "PhaseOutput",
    "ProblemBlock",
    "RunArtifact",
    "RunConfig",
    "SolutionState",
    "ExplorationTrajectory",
    "AnalyzerPrepared",
    "EntropicSegregator",
    "VerticalTopology",
    "DescentVector",
    "ConvergenceProtocol",
    "ComplexityScorer",
    "SemanticGradient",
    "ConsistencyEnforcer",
    "CPPTAITraslocatore",
    "ResponsibleAIAuditor",
    "run_v2",
    "CoTBaseline",
    "ToTBaseline",
    "GoTBaseline",
    "ReActBaseline",
]
