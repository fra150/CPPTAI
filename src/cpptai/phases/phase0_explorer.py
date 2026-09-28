"""Phase 0: Explorer — diffusion reasoning generates interpretations."""
from __future__ import annotations
import time
from ..exploration import ExplorerEngine
from ..types import PhaseOutput, RunConfig, SolutionState

def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    engine = ExplorerEngine(
        num_trajectories=config.explorer_num_trajectories,
        noise_level=config.explorer_noise_level,
        temperature=config.explorer_temperature,
        denoising_steps=config.explorer_denoising_steps,
        seed=config.seed,
        adaptive=config.explorer_adaptive,
        max_workers=config.explorer_max_workers,
        parallel_llm=config.explorer_parallel_llm,
        cache_results=config.explorer_cache_results,
        diversity_weight=config.explorer_diversity_weight,
    )
    trajectories = engine.explore(state.problem)
    dt = time.perf_counter() - t0
    state.explorer_trajectories = trajectories
    state.extra["explorer_best"] = engine.select_best(trajectories).to_summary_dict()
    return (
        state,
        PhaseOutput(
            name="phase0_explorer",
            input={"problem": state.problem},
            output={
                "num_trajectories": len(trajectories),
                "trajectories": [t.to_summary_dict() for t in trajectories],
                "diversity_score": round(
                    (max(t.novelty_score for t in trajectories) if trajectories else 0.0), 4
                ),
            },
            duration_sec=dt,
        ),
    )
