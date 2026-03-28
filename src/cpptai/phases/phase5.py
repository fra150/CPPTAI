"""Phase V: Presentation."""

from __future__ import annotations

import time

from ..presentation import arrange_solution_simple
from ..types import PhaseOutput, RunConfig, SolutionState


def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    descent = state.extra.get("descent_result") or {}
    external = state.extra.get("external") or {}
    raw = (descent.get("final_answer", "") + "\n" + external.get("external_synthesis", "")).strip()
    arranged = arrange_solution_simple(raw, context="technical")
    dt = time.perf_counter() - t0
    state.extra["final_answer"] = raw
    state.extra["final_arranged"] = arranged
    return (
        state,
        PhaseOutput(
            name="phase5_presentation",
            input={"context": "technical"},
            output={"final_answer": raw, "final_arranged": arranged},
            duration_sec=dt,
        ),
    )
