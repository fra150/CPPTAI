"""Phase III: Cognitive Descent."""

from __future__ import annotations

import time

from ..core import DescentVector
from ..types import PhaseOutput, RunConfig, SolutionState


def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    descent = DescentVector()
    initial_context = {
        "problem": state.problem,
        "block_solutions": state.block_solutions,
        "building_height": state.building_height,
        "blocks": state.blocks,
    }
    descent_result = descent.cognitive_descent(state.building_height, initial_context)
    dt = time.perf_counter() - t0
    state.extra["descent_result"] = descent_result
    return (
        state,
        PhaseOutput(
            name="phase3_cognitive_descent",
            input={"building_height": state.building_height},
            output={
                "final_answer": descent_result.get("final_answer", ""),
                "counterfactual_summary": descent_result.get("counterfactual_summary", ""),
                "attribution_explanation": descent_result.get("attribution_explanation", ""),
            },
            duration_sec=dt,
        ),
    )
