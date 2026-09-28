"""Type definitions and core data structures for the CPPTAI framework.

All names and docstrings are in English, per user request.
"""

import os
import uuid
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
class ExplorationTrajectory:
    """Singola traiettoria di esplorazione generata dall'Explorer (Phase 0).

    Ogni traiettoria rappresenta un'interpretazione/framing del problema
    originale, generata con un diverso 'seed di rumore' e progressivamente
    raffinata attraverso denoising steps.
    """

    id: str
    interpretation: str
    confidence: float
    reasoning_path: List[str]
    noise_seed: int
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "interpretation": self.interpretation[:200],
            "confidence": round(self.confidence, 4),
            "reasoning_steps": len(self.reasoning_path),
            "novelty": round(self.novelty_score, 4),
        }


@dataclass
class AnalyzerPrepared:
    """Output della fase Analyzer (Phase 0.5) — problema arricchito per Phase I.

    L'Analyzer prende le traiettorie dell'Explorer, seleziona le migliori,
    e produce un problema arricchito che diventa l'input di Phase I.
    """

    enriched_problem: str
    top_interpretations: List[str]
    aggregate_confidence: float
    reasoning_summary: str
    trajectories_used: int
    patterns_identified: List[str] = field(default_factory=list)
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
    domain: str = ""
    problem_type: str = ""
    # --- Explorer + Analyzer options (default: disabled for backward compat) ---
    explorer_enabled: bool = False
    analyzer_enabled: bool = False
    explorer_num_trajectories: int = 5
    explorer_batch_size: int = 1
    explorer_max_workers: int = 4
    explorer_adaptive: bool = True
    explorer_parallel_llm: bool = False
    explorer_cache_results: bool = True
    explorer_diversity_weight: float = 0.6
    explorer_noise_level: float = 0.6
    explorer_temperature: float = 0.8
    explorer_denoising_steps: int = 3
    analyzer_ensemble_size: int = 3
    analyzer_coherence_threshold: float = 0.5

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
        explorer_enabled = os.getenv("CPPTAI_EXPLORER_ENABLED", "0") == "1"
        analyzer_enabled = os.getenv("CPPTAI_ANALYZER_ENABLED", "0") == "1"
        explorer_num_trajectories = int(os.getenv("CPPTAI_EXPLORER_TRAJECTORIES", "5"))
        explorer_batch_size = int(os.getenv("CPPTAI_EXPLORER_BATCH_SIZE", "1"))
        explorer_max_workers = int(os.getenv("CPPTAI_EXPLORER_MAX_WORKERS", "4"))
        explorer_adaptive = os.getenv("CPPTAI_EXPLORER_ADAPTIVE", "1") == "1"
        explorer_parallel_llm = os.getenv("CPPTAI_EXPLORER_PARALLEL_LLM", "0") == "1"
        explorer_cache_results = os.getenv("CPPTAI_EXPLORER_CACHE", "1") == "1"
        explorer_diversity_weight = float(os.getenv("CPPTAI_EXPLORER_DIVERSITY_WEIGHT", "0.6"))
        explorer_noise_level = float(os.getenv("CPPTAI_EXPLORER_NOISE", "0.6"))
        explorer_temperature = float(os.getenv("CPPTAI_EXPLORER_TEMP", "0.8"))
        explorer_denoising_steps = int(os.getenv("CPPTAI_EXPLORER_DENOISE_STEPS", "3"))
        analyzer_ensemble_size = int(os.getenv("CPPTAI_ANALYZER_ENSEMBLE", "3"))
        analyzer_coherence_threshold = float(os.getenv("CPPTAI_ANALYZER_THRESHOLD", "0.5"))
        return RunConfig(
            seed=seed,
            external_enabled=external_enabled,
            cache_mode=cache_mode,
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            model_name=model_name,
            max_iterations=max_iterations,
            explorer_enabled=explorer_enabled,
            analyzer_enabled=analyzer_enabled,
            explorer_num_trajectories=explorer_num_trajectories,
            explorer_batch_size=explorer_batch_size,
            explorer_max_workers=explorer_max_workers,
            explorer_adaptive=explorer_adaptive,
            explorer_parallel_llm=explorer_parallel_llm,
            explorer_cache_results=explorer_cache_results,
            explorer_diversity_weight=explorer_diversity_weight,
            explorer_noise_level=explorer_noise_level,
            explorer_temperature=explorer_temperature,
            explorer_denoising_steps=explorer_denoising_steps,
            analyzer_ensemble_size=analyzer_ensemble_size,
            analyzer_coherence_threshold=analyzer_coherence_threshold,
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
    # --- Explorer + Analyzer campi ---
    explorer_trajectories_count: Optional[int] = None
    analyzer_confidence: Optional[float] = None
    exploration_data: Optional[Dict[str, Any]] = None


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
    # --- Explorer + Analyzer campi ---
    explorer_trajectories: List[ExplorationTrajectory] = field(default_factory=list)
    analyzer_prepared: Optional[AnalyzerPrepared] = None
    exploration_enabled: bool = True

