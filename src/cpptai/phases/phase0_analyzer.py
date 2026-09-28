"""Phase 0.5: Analyzer — select and prepare best trajectories for CPPTAI Phase I."""
from __future__ import annotations
import time
from ..exploration import TrajectoryAnalyzer
from ..types import PhaseOutput, RunConfig, SolutionState

def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    analyzer = TrajectoryAnalyzer(
        ensemble_size=config.analyzer_ensemble_size,
        coherence_threshold=config.analyzer_coherence_threshold,
    )
    trajectories = state.explorer_trajectories
    prepared = analyzer.analyze(trajectories, state.problem)
    dt = time.perf_counter() - t0
    state.analyzer_prepared = prepared
    state.problem = prepared.enriched_problem
    state.extra["analyzer_confidence"] = prepared.aggregate_confidence
    state.extra["analyzer_top_interpretations"] = prepared.top_interpretations
    return (
        state,
        PhaseOutput(
            name="phase0_analyzer",
            input={
                "num_trajectories": len(trajectories),
                "ensemble_size": config.analyzer_ensemble_size,
            },
            output={
                "enriched_problem_length": len(prepared.enriched_problem),
                "top_interpretations": prepared.top_interpretations,
                "aggregate_confidence": prepared.aggregate_confidence,
                "trajectories_used": prepared.trajectories_used,
                "patterns": prepared.patterns_identified,
            },
            duration_sec=dt,
        ),
    )
