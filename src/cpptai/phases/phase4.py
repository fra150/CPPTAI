"""Phase IV: External Convergence."""

from __future__ import annotations

import time

from ..core import ConvergenceProtocol
from ..types import PhaseOutput, RunConfig, SolutionState


def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    conv = ConvergenceProtocol()
    external = {"external_synthesis": "", "responses": {}, "confidence": 0.0}
    if config.external_enabled:
        external = conv.convene_meeting({"problem": state.problem})
    dt = time.perf_counter() - t0
    state.extra["external"] = external
    return (
        state,
        PhaseOutput(
            name="phase4_external_convergence",
            input={"external_enabled": config.external_enabled},
            output={"confidence": external.get("confidence", 0.0), "external_synthesis": external.get("external_synthesis", "")},
            duration_sec=dt,
        ),
    )
