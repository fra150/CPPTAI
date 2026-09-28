"""Phase II: Vertical Topology."""

from __future__ import annotations

import time

from ..core import VerticalTopology
from ..types import PhaseOutput, RunConfig, SolutionState


def run(state: SolutionState, config: RunConfig) -> tuple[SolutionState, PhaseOutput]:
    t0 = time.perf_counter()
    topology = VerticalTopology()
    building_height = topology.calculate_building_height(state.blocks)
    topology.assign_floors(state.blocks, building_height)
    state.building_height = building_height
    dt = time.perf_counter() - t0
    return (
        state,
        PhaseOutput(
            name="phase2_vertical_topology",
            input={"block_count": len(state.blocks)},
            output={"building_height": building_height, "floors": {b.id: b.floor_index for b in state.blocks}},
            duration_sec=dt,
        ),
    )
