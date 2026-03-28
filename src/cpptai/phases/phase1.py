"""Phase I: Entropic Segregation."""

from __future__ import annotations

import time

from ..core import EntropicSegregator
from ..types import PhaseOutput, RunConfig, SolutionState


def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    segregator = EntropicSegregator()
    blocks = segregator.segregate(state.problem)
    block_solutions = [segregator.solve_linear_cot(b) for b in blocks]
    state.blocks = blocks
    state.block_solutions = block_solutions
    dt = time.perf_counter() - t0
    return (
        state,
        PhaseOutput(
            name="phase1_entropic_segregation",
            input={"problem": state.problem},
            output={"blocks": [b.id for b in blocks], "block_solutions": block_solutions},
            duration_sec=dt,
        ),
    )
