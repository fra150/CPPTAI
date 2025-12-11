"""CPPTAI package initializer.

Exports the main orchestrator and core types for convenience.
"""
from .types import DifficultyLevel, ProblemBlock
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

__all__ = [
    "DifficultyLevel",
    "ProblemBlock",
    "EntropicSegregator",
    "VerticalTopology",
    "DescentVector",
    "ConvergenceProtocol",
    "ComplexityScorer",
    "SemanticGradient",
    "ConsistencyEnforcer",
    "CPPTAITraslocatore",
]

