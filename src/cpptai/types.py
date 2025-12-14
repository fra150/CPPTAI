"""Type definitions and core data structures for the CPPTAI framework.

All names and docstrings are in English, per user request.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class DifficultyLevel(Enum):
    """Discrete difficulty levels used to rank problem blocks."""

    IMPOSSIBLE = 5
    HARD = 4
    MEDIUM = 3
    NORMAL = 2
    EASY = 1
    TRIVIAL = 0


@dataclass
class ProblemBlock:
    """Atomic unit extracted from a complex problem statement.

    Attributes:
        id: Stable identifier for the block.
        content: Raw text content of the block.
        difficulty: Coarse difficulty level for sorting and reporting.
        complexity_score: Continuous [0, 1] score estimating inherent complexity.
        solution_probability: Continuous [0, 1] score estimating solvability.
        improbability: Continuous [0, 1] score = 1 - solution_probability.
        floor_index: Integer floor assigned in Vertical Topology (Phase II).
        dependencies: IDs of other blocks this block depends on.
    """

    id: str
    content: str
    difficulty: DifficultyLevel
    complexity_score: float
    solution_probability: float
    improbability: float
    floor_index: int = 0
    dependencies: List[str] = field(default_factory=list)
    influence_score: float = 0.0

