"""CLI entrypoint for the CPPTAI framework.
Runs a sample complex problem through the end-to-end pipeline and reports the
final answer along with the locations of persisted artifacts.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional
from cpptai.core import CPPTAITraslocatore
from cpptai.pipeline_v2 import run as run_v2
from cpptai.types import RunConfig

def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="cpptai", add_help=True)
    parser.add_argument("--v2", action="store_true", help="Run the v2 pipeline runner")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--no-benchmark", action="store_true", help="Disable benchmark execution")
    parser.add_argument("--offline", action="store_true", help="Disable external calls and use offline mode")
    parser.add_argument("--output-dir", default=".", help="Output directory for v2 run artifacts")
    ns = parser.parse_args(args or sys.argv[1:])

    if ns.offline:
        os.environ["BENCH_DISABLE_EXTERNAL"] = "1"
        os.environ["CPPTAI_DISABLE_EXTERNAL"] = "1"
        os.environ.setdefault("CPPTAI_CACHE_MODE", "offline")

    problem = (
        "How can we address the global energy crisis considering: "
        "1) limits of renewables, 2) nuclear costs, 3) fossil dependency, "
        "4) geopolitical factors, 5) a just transition for workers?"
    )

    if ns.v2:
        cfg = RunConfig.from_env()
        cfg.output_dir = ns.output_dir
        cfg.external_enabled = not ns.offline and cfg.external_enabled
        result, artifact = run_v2(problem, cfg)
        print("\nFinal Answer:\n" + result.get("final_answer", "Undetermined"))
        arranged = result.get("final_arranged")
        if arranged:
            print("\nArranged (Phase V):\n" + arranged)
        print(f"V2 artifact saved to: {os.path.join(ns.output_dir, f'run_{artifact.run_id}.json')}")
    else:
        traslocatore = CPPTAITraslocatore(enable_phase_iv=not ns.offline)
        result = traslocatore.solve(problem)
        print("\nFinal Answer:\n" + result.get("final_answer", "Undetermined"))
        arranged = result.get("final_arranged")
        if arranged:
            print("\nArranged (Phase V):\n" + arranged)
        print("Artifacts saved to: memoria.json, ragionamenti.csv")

    run_bench = ns.benchmark or (not ns.no_benchmark)
    if run_bench:
        from cpptai.benchmarks import run_benchmarks

        print("\nRunning benchmarks…")
        records, summary = run_benchmarks()
        print("Summary (accuracy, diversity, error_rate, time_sec, tokens):")
        for method, stats in summary.items():
            print(f"  {method}: {stats}")
        print("Benchmark artifacts saved to: benchmarks.csv, benchmarks.json")

if __name__ == "__main__":
    main()
