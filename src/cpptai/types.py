"""Type definitions and core data structures for the CPPTAI framework.

All names and docstrings are in English, per user request.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


@dataclass
class BenchmarkItem:
    """Canonical benchmark item used by runners and evaluators."""

    id: str
    prompt: str
    expected: List[str]
    dataset: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Run configuration for reproducible execution."""

    seed: int = 0
    external_enabled: bool = True
    cache_mode: str = "online"
    output_dir: str = "."
    benchmark_name: str = "mixed"
    model_name: str = "DeepSeek-V3.2-Exp"
    max_iterations: int = 100

    @staticmethod
    def from_env() -> "RunConfig":
        external_enabled = (os.getenv("BENCH_DISABLE_EXTERNAL", "0") != "1") and (
            os.getenv("CPPTAI_DISABLE_EXTERNAL", "0") != "1"
        )
        cache_mode = os.getenv("CPPTAI_CACHE_MODE", "online")
        seed = int(os.getenv("CPPTAI_SEED", "0"))
        output_dir = os.getenv("CPPTAI_OUTPUT_DIR", ".")
        benchmark_name = os.getenv("CPPTAI_BENCHMARK", "mixed")
        model_name = os.getenv("CPPTAI_MODEL", "DeepSeek-V3.2-Exp")
        max_iterations = int(os.getenv("CPPTAI_MAX_ITERS", "100"))
        return RunConfig(
            seed=seed,
            external_enabled=external_enabled,
            cache_mode=cache_mode,
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            model_name=model_name,
            max_iterations=max_iterations,
        )


@dataclass
class PhaseOutput:
    """Structured phase output for artifacts."""

    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    decisions: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_sec: float = 0.0


@dataclass
class RunArtifact:
    """Canonical artifact schema for reproducibility and auditability."""

    run_id: str
    git_commit: str
    timestamp: str
    benchmark_name: str
    task_id: str
    model_name: str
    seed: int
    external_enabled: bool
    cache_mode: str
    phase_outputs: List[PhaseOutput]
    entropy_by_phase: Dict[str, float]
    final_answer: str
    verification_result: Dict[str, Any]
    runtime_seconds: float
    token_counts: Dict[str, int]
    provenance: Dict[str, Any] = field(default_factory=dict)
    failure_phase: Optional[str] = None
    failure_reason: Optional[str] = None
    fallback_triggered: bool = False


@dataclass
class SolutionState:
    """Canonical solution state tracked across phases."""

    problem: str
    blocks: List[ProblemBlock] = field(default_factory=list)
    building_height: int = 1
    block_solutions: List[Dict[str, Any]] = field(default_factory=list)
    coherence: float = 0.2
    completeness: float = 0.2
    confidence: float = 0.2
    extra: Dict[str, Any] = field(default_factory=dict)

