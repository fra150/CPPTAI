"""CPPTAI v2 pipeline runner (reproducible, structured artifacts)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .phases import (
    run_phase0_explorer, run_phase0_analyzer,
    run_phase1, run_phase2, run_phase3, run_phase4, run_phase5,
)
from .types import DifficultyLevel, PhaseOutput, ProblemBlock, RunArtifact, RunConfig, SolutionState
from .responsible_ai import ResponsibleAIAuditor


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fallback_blocks(problem: str) -> List[ProblemBlock]:
    return [
        ProblemBlock(
            id="B1",
            content=problem,
            difficulty=DifficultyLevel.NORMAL,
            complexity_score=0.5,
            solution_probability=0.5,
            improbability=0.5,
            floor_index=0,
            dependencies=[],
        )
    ]


def run(problem: str, config: Optional[RunConfig] = None) -> Tuple[Dict[str, Any], RunArtifact]:
    config = config or RunConfig.from_env()
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    state = SolutionState(problem=problem)
    phase_outputs: List[PhaseOutput] = []
    entropy_by_phase: Dict[str, float] = {}
    failure_phase: Optional[str] = None
    failure_reason: Optional[str] = None
    fallback_triggered = False

    try:
        # === Fase 0: Explorer ===
        if config.explorer_enabled and os.getenv("CPPTAI_DISABLE_PHASE0", "0") != "1":
            state, p0 = run_phase0_explorer(state, config)
            phase_outputs.append(p0)
            entropy_by_phase[p0.name] = float(len(state.explorer_trajectories)) / max(
                1.0, config.explorer_num_trajectories
            )

        # === Fase 0.5: Analyzer ===
        if config.analyzer_enabled and os.getenv("CPPTAI_DISABLE_PHASE0B", "0") != "1":
            state, p0b = run_phase0_analyzer(state, config)
            phase_outputs.append(p0b)
            entropy_by_phase[p0b.name] = float(p0b.output.get("aggregate_confidence", 0.0))

        # === Fase 1: Entropic Segregation ===
        if os.getenv("CPPTAI_DISABLE_PHASE1", "0") == "1":
            state.blocks = _fallback_blocks(problem)
        else:
            state, p1 = run_phase1(state, config)
            phase_outputs.append(p1)
            entropy_by_phase[p1.name] = float(len(state.blocks)) / max(1.0, len(problem.split()))

        if os.getenv("CPPTAI_DISABLE_PHASE2", "0") != "1":
            state, p2 = run_phase2(state, config)
            phase_outputs.append(p2)
            entropy_by_phase[p2.name] = float(state.building_height) / max(1.0, len(state.blocks))

        if os.getenv("CPPTAI_DISABLE_PHASE3", "0") != "1":
            state, p3 = run_phase3(state, config)
            phase_outputs.append(p3)
            entropy_by_phase[p3.name] = float(len((state.extra.get("descent_result") or {}).get("descent_log", []))) / max(
                1.0, state.building_height
            )

        if os.getenv("CPPTAI_DISABLE_PHASE4", "0") != "1":
            state, p4 = run_phase4(state, config)
            phase_outputs.append(p4)
            entropy_by_phase[p4.name] = float(p4.output.get("confidence", 0.0))

        if os.getenv("CPPTAI_DISABLE_PHASE5", "0") != "1":
            state, p5 = run_phase5(state, config)
            phase_outputs.append(p5)
            entropy_by_phase[p5.name] = float(len((state.extra.get("final_answer") or "").split())) / 100.0

        final_answer = str(state.extra.get("final_answer") or "")
        final_arranged = str(state.extra.get("final_arranged") or "")

        auditor = ResponsibleAIAuditor()
        audit = auditor.audit_bias_detection(final_arranged)

        result: Dict[str, Any] = {
            "final_answer": final_answer,
            "final_arranged": final_arranged,
            "responsible_ai_audit": audit,
            "phase_outputs": [asdict(p) for p in phase_outputs],
            "entropy_by_phase": entropy_by_phase,
        }

        runtime = time.perf_counter() - t0
        artifact = RunArtifact(
            run_id=run_id,
            git_commit=_git_commit(),
            timestamp=_utc_now(),
            benchmark_name=config.benchmark_name,
            task_id="cli_default",
            model_name=config.model_name,
            seed=config.seed,
            external_enabled=config.external_enabled,
            cache_mode=config.cache_mode,
            phase_outputs=phase_outputs,
            entropy_by_phase=entropy_by_phase,
            final_answer=final_answer,
            verification_result={"passed": True, "score": None, "verifier_type": "none"},
            runtime_seconds=float(runtime),
            token_counts={"final_answer_tokens": len(final_answer.split())},
            provenance={"external": state.extra.get("external") or {}},
            failure_phase=failure_phase,
            failure_reason=failure_reason,
            fallback_triggered=fallback_triggered,
            explorer_trajectories_count=len(state.explorer_trajectories) if state.explorer_trajectories else None,
            analyzer_confidence=state.analyzer_prepared.aggregate_confidence if state.analyzer_prepared else None,
            exploration_data={
                "num_trajectories": len(state.explorer_trajectories) if state.explorer_trajectories else 0,
                "aggregate_confidence": state.analyzer_prepared.aggregate_confidence if state.analyzer_prepared else 0.0,
                "patterns": state.analyzer_prepared.patterns_identified if state.analyzer_prepared else [],
            } if (state.explorer_trajectories or state.analyzer_prepared) else None,
        )
        _persist_artifact(artifact, config.output_dir)
        return result, artifact
    except Exception as e:
        runtime = time.perf_counter() - t0
        failure_phase = failure_phase or "unknown"
        failure_reason = failure_reason or str(e)
        artifact = RunArtifact(
            run_id=run_id,
            git_commit=_git_commit(),
            timestamp=_utc_now(),
            benchmark_name=config.benchmark_name,
            task_id="cli_default",
            model_name=config.model_name,
            seed=config.seed,
            external_enabled=config.external_enabled,
            cache_mode=config.cache_mode,
            phase_outputs=phase_outputs,
            entropy_by_phase=entropy_by_phase,
            final_answer="",
            verification_result={"passed": False, "score": 0.0, "verifier_type": "exception", "notes": failure_reason},
            runtime_seconds=float(runtime),
            token_counts={},
            provenance={},
            failure_phase=failure_phase,
            failure_reason=failure_reason,
            fallback_triggered=True,
            explorer_trajectories_count=len(state.explorer_trajectories) if state.explorer_trajectories else None,
            analyzer_confidence=state.analyzer_prepared.aggregate_confidence if state.analyzer_prepared else None,
        )
        _persist_artifact(artifact, config.output_dir)
        raise


def solve_with_cpptai(problem: str, config: Optional[RunConfig] = None) -> Dict[str, Any]:
    """Solve a problem using the CPPTAI v2 pipeline.

    Wraps the pipeline run() function for use by benchmark scripts.

    Args:
        problem: The problem statement to solve.
        config: Optional RunConfig. Defaults to environment-based config.

    Returns:
        Dict with 'answer' (the final answer) and all pipeline result fields.
    """
    try:
        result, artifact = run(problem, config)
        result["answer"] = result.get("final_answer", "")
        return result
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("solve_with_cpptai failed: %s", e, exc_info=True)
        return {"answer": "", "final_answer": "", "error": str(e)}


def _persist_artifact(artifact: RunArtifact, output_dir: str) -> None:
    out = _ensure_dir(output_dir)
    path = out / f"run_{artifact.run_id}.json"
    payload = asdict(artifact)
    payload["phase_outputs"] = [asdict(p) for p in artifact.phase_outputs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
