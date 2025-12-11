"""CLI entrypoint for the CPPTAI framework.
Runs a sample complex problem through the end-to-end pipeline and reports the
final answer along with the locations of persisted artifacts.
"""

from __future__ import annotations
import sys
from typing import Optional
from cpptai.core import CPPTAITraslocatore
from cpptai.benchmarks import run_benchmarks

def main(args: Optional[list[str]] = None) -> None:
    args = args or sys.argv[1:]
    traslocatore = CPPTAITraslocatore()
    problem = (
        "How can we address the global energy crisis considering: "
        "1) limits of renewables, 2) nuclear costs, 3) fossil dependency, "
        "4) geopolitical factors, 5) a just transition for workers?"
    )

    result = traslocatore.solve(problem)
    print("\nFinal Answer:\n" + result.get("final_answer", "Undetermined"))
    arranged = result.get("final_arranged")
    if arranged:
        print("\nArranged (Phase V):\n" + arranged)
    print("Artifacts saved to: memoria.json, ragionamenti.csv")

    print("\nRunning benchmarks…")
    records, summary = run_benchmarks()
    print("Summary (accuracy, diversity, error_rate, time_sec, tokens):")
    for method, stats in summary.items():
        print(f"  {method}: {stats}")
    print("Benchmark artifacts saved to: benchmarks.csv, benchmarks.json")

if __name__ == "__main__":
    main()
