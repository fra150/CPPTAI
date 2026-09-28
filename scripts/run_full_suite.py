"""Full benchmark suite — GSM8K complete, MATH, HumanEval, ablation studies.
Runs all configured benchmarks and generates comprehensive reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("src"))

from cpptai.types import RunConfig, RunArtifact, ExplorationTrajectory
from cpptai.pipeline_v2 import run as run_v2
from cpptai.core import CPPTAITraslocatore
from cpptai.exploration import ExplorerEngine, TrajectoryAnalyzer
from cpptai.datasets import DatasetLoader
from cpptai.benchmarks import (
    build_problems, run_benchmarks, gsm8k_accuracy, rubric_accuracy,
    compute_pass_at_k, category_breakdown,
)
from cpptai.humaneval_executor import verify_humaneval, is_valid_python


BANNER = """
===============================================
    CPPTAI FULL BENCHMARK SUITE
    GSM8K (1-1319) | MATH (100) | HumanEval
===============================================
"""


class BenchmarkOrchestrator:
    """Orchestrates multi-benchmark runs with progress tracking."""

    def __init__(self, output_dir: str = "benchmarks", workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.results: List[Dict] = []
        self.start_time = time.perf_counter()

    def run_gsm8k(self, n: int = 1319, with_explorer: bool = False, method: str = "bypass") -> Dict:
        """Run GSM8K benchmark on n problems via real API call.

        method:
          "bypass"   -> orchestrator.solve_gsm8k (single DeepSeek call; no phases)
          "pipeline" -> orchestrator.solve (full CPPTAI cognitive descent pipeline)
        """
        method_label = ("CPPTAI_EA" if with_explorer else
                        ("CPPTAI_pipeline" if method == "pipeline" else "CPPTAI"))
        print(f"\nGSM8K ({n} problems, explorer={with_explorer}, method={method})")
        loader = DatasetLoader()
        problems = loader.load_gsm8k(n=n)
        
        orchestrator = CPPTAITraslocatore(enable_phase_iv=False)
        
        # Prepara Explorer se richiesto
        explorer = None
        analyzer = None
        if with_explorer:
            explorer = ExplorerEngine(num_trajectories=5, seed=0)
            analyzer = TrajectoryAnalyzer(ensemble_size=3)
        
        records = []
        t0 = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for problem in problems:
                future = executor.submit(
                    self._solve_gsm8k_single, problem, orchestrator, explorer, analyzer, method
                )
                futures[future] = problem
            
            for i, future in enumerate(as_completed(futures)):
                problem = futures[future]
                try:
                    answer, runtime = future.result()
                    pid = problem["id"]
                    expected = problem.get("expected", [])
                    acc = gsm8k_accuracy(answer, expected)
                    cat = problem.get("metadata", {}).get("category", "unknown")
                    
                    records.append({
                        "problem_id": pid,
                        "category": cat,
                        "accuracy": round(acc, 3),
                        "time_sec": round(runtime, 3),
                        "method": method_label,
                        "dataset": "gsm8k",
                    })
                except Exception as e:
                    records.append({
                        "problem_id": problem.get("id", "?"),
                        "category": "error",
                        "accuracy": 0.0,
                        "time_sec": 0.0,
                        "method": method_label,
                        "dataset": "gsm8k",
                    })
                
                if (i + 1) % 5 == 0 or (i + 1) == n:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    correct_sofar = sum(1 for r in records if r.get("accuracy", 0) >= 1.0)
                    acc_sofar = correct_sofar / max(1, len(records))
                    print(f"  Progress: {i+1}/{n} | acc={acc_sofar:.3f} | {rate:.1f}/sec")
        
        total_time = time.perf_counter() - t0
        accuracy = compute_pass_at_k(records)
        categories = category_breakdown(records)
        
        summary = {
            "dataset": "gsm8k",
            "method": method_label,
            "n": len(records),
            "accuracy": accuracy,
            "total_time_sec": round(total_time, 1),
            "problems_per_sec": round(len(records) / total_time, 2),
            "category_breakdown": categories,
        }
        
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) — Time: {total_time:.0f}s")
        
        self.results.extend(records)
        return summary

    def _solve_gsm8k_single(
        self, problem: Dict, orchestrator: CPPTAITraslocatore,
        explorer: Optional[ExplorerEngine], analyzer: Optional[TrajectoryAnalyzer],
        method: str = "bypass",
    ) -> Tuple[str, float]:
        """Risolvi un singolo problema GSM8K, opzionalmente con Explorer."""
        t0 = time.perf_counter()
        prompt = problem["prompt"]

        if explorer and analyzer:
            # Explorer + Analyzer preprocessing
            trajectories = explorer.explore(prompt)
            prepared = analyzer.analyze(trajectories, prompt)
            prompt = prepared.enriched_problem

        if method == "pipeline":
            # Full CPPTAI cognitive pipeline. Fresh orchestrator per call keeps
            # the per-run long-term-memory / archive state thread-safe.
            orch = CPPTAITraslocatore(enable_phase_iv=False)
            result = orch.solve(prompt)
        else:
            result = orchestrator.solve_gsm8k(prompt)
        return result.get("final_answer", ""), time.perf_counter() - t0

    def run_humaneval(self, n: int = 164) -> Dict:
        """Run HumanEval with real execution verification."""
        print(f"\nHumanEval ({n} problems, real execution)")
        loader = DatasetLoader()
        problems = loader.load_humaneval(n=n)

        cfg = RunConfig(
            seed=0, external_enabled=False, cache_mode="offline",
            benchmark_name="humaneval",
        )

        records = []
        t0 = time.perf_counter()

        for i, problem in enumerate(problems):
            result, _ = run_v2(problem["prompt"], cfg)
            generated = result.get("final_answer", "")

            test_code = problem.get("expected", [""])[0] if problem.get("expected") else ""

            verification = verify_humaneval(generated, test_code, "check")

            records.append({
                "problem_id": problem["id"],
                "accuracy": 1.0 if verification["passed"] else 0.0,
                "time_sec": verification.get("timing", 0),
                "method": "CPPTAI",
                "dataset": "humaneval",
                "execution_passed": verification["passed"],
            })

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{n}")

        total_time = time.perf_counter() - t0
        accuracy = compute_pass_at_k(records)

        summary = {
            "dataset": "humaneval",
            "n": len(records),
            "accuracy": accuracy,
            "total_time_sec": round(total_time, 1),
        }

        print(f"  Pass@1: {accuracy:.3f} ({accuracy*100:.1f}%)")
        self.results.extend(records)
        return summary

    def run_math(self, n: int = 100) -> Dict:
        """Run MATH benchmark."""
        print(f"\nMATH ({n} problems)")
        loader = DatasetLoader()
        problems = loader.load_math(n=n)

        cfg = RunConfig(
            seed=0, external_enabled=False, cache_mode="offline",
            benchmark_name="math",
        )

        records = []
        t0 = time.perf_counter()

        for i, problem in enumerate(problems):
            result, _ = run_v2(problem["prompt"], cfg)
            acc = gsm8k_accuracy(result.get("final_answer", ""), problem.get("expected", []))
            records.append({
                "problem_id": problem["id"],
                "accuracy": round(acc, 3),
                "method": "CPPTAI",
                "dataset": "math",
            })
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{n}")

        accuracy = compute_pass_at_k(records)
        print(f"  Accuracy: {accuracy:.3f}")
        self.results.extend(records)
        return {"dataset": "math", "n": len(records), "accuracy": accuracy}

    def run_ablation_study(self, problems_list: List[Dict]) -> Dict:
        """Run ablation: CPPTAI vs CPPTAI+EA vs CPPTAI_no_I vs CPPTAI_no_IV."""
        print(f"\nAblation Study ({len(problems_list)} problems)")
        configs = [
            ("CPPTAI", RunConfig(seed=0, external_enabled=False, cache_mode="offline")),
            ("CPPTAI_EA", RunConfig(seed=0, external_enabled=False, cache_mode="offline",
                                    explorer_enabled=True, analyzer_enabled=True, explorer_num_trajectories=3)),
        ]

        records = []
        t0 = time.perf_counter()

        for method_name, cfg in configs:
            for problem in problems_list[:50]:
                result, _ = run_v2(problem["prompt"], cfg)
                acc = gsm8k_accuracy(result.get("final_answer", ""), problem.get("expected", []))
                records.append({
                    "problem_id": problem["id"],
                    "method": method_name,
                    "accuracy": round(acc, 3),
                    "dataset": "ablation",
                })

        self.results.extend(records)
        return {"ablation": "50 problems x 2 configs"}

    def generate_reports(self):
        """Generate comprehensive reports from all results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = self.output_dir / f"full_benchmarks_{timestamp}.csv"
        if self.results:
            fieldnames = list(self.results[0].keys())
            # Assicura che tutti i record abbiamo gli stessi campi
            for r in self.results:
                for k in fieldnames:
                    r.setdefault(k, "")
                for k in list(r.keys()):
                    if k not in fieldnames:
                        r.pop(k, None)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\nCSV: {csv_path}")

        summary = {
            "timestamp": timestamp,
            "total_problems": len(self.results),
            "average_accuracy": round(
                sum(r.get("accuracy", 0) for r in self.results) / max(1, len(self.results)), 4
            ),
            "by_dataset": self._group_by_dataset(),
            "by_method": self._group_by_method(),
        }
        json_path = self.output_dir / f"summary_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON: {json_path}")

    def _group_by_dataset(self) -> Dict:
        groups = {}
        for r in self.results:
            ds = r.get("dataset", "unknown")
            if ds not in groups:
                groups[ds] = {"n": 0, "correct": 0}
            groups[ds]["n"] += 1
            if r.get("accuracy", 0) >= 1.0:
                groups[ds]["correct"] += 1
        for ds in groups:
            groups[ds]["accuracy"] = round(groups[ds]["correct"] / max(1, groups[ds]["n"]), 3)
        return groups

    def _group_by_method(self) -> Dict:
        groups = {}
        for r in self.results:
            m = r.get("method", "unknown")
            if m not in groups:
                groups[m] = {"n": 0, "correct": 0}
            groups[m]["n"] += 1
            if r.get("accuracy", 0) >= 1.0:
                groups[m]["correct"] += 1
        for m in groups:
            groups[m]["accuracy"] = round(groups[m]["correct"] / max(1, groups[m]["n"]), 3)
        return groups


def main():
    print(BANNER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmarks/full_suite")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gsm8k", type=int, default=50, help="GSM8K problems (0=skip, 1319=all)")
    parser.add_argument("--math", type=int, default=20)
    parser.add_argument("--humaneval", type=int, default=10)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--explorer", action="store_true", help="Run GSM8K with Explorer")
    parser.add_argument("--compare", action="store_true", help="Run GSM8K with AND without Explorer (2x)")
    parser.add_argument("--gsm8k-method", choices=["bypass", "pipeline"], default="bypass",
                        help="GSM8K solver: bypass (single DeepSeek call) or pipeline (full CPPTAI)")
    parser.add_argument("--compare-pipeline", action="store_true",
                        help="Run GSM8K with bypass AND full pipeline (2x) and compare")
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(output_dir=args.output_dir, workers=args.workers)

    if args.compare_pipeline:
        s1 = orchestrator.run_gsm8k(n=args.gsm8k, method="bypass")
        s2 = orchestrator.run_gsm8k(n=args.gsm8k, method="pipeline")
        delta = (s2["accuracy"] - s1["accuracy"]) * 100
        print(f"\n[PIPELINE COMPARE] n={args.gsm8k} | "
              f"bypass(solve_gsm8k)={s1['accuracy']*100:.1f}% "
              f"vs pipeline(solve)={s2['accuracy']*100:.1f}% | delta={delta:+.1f}pp")
    elif args.compare:
        s1 = orchestrator.run_gsm8k(n=args.gsm8k, with_explorer=False)
        s2 = orchestrator.run_gsm8k(n=args.gsm8k, with_explorer=True)
        print(f"\n[ABLATION] without={s1['accuracy']:.3f} vs with={s2['accuracy']:.3f}")
    elif args.explorer:
        orchestrator.run_gsm8k(n=args.gsm8k, with_explorer=True)
    else:
        if args.gsm8k > 0:
            orchestrator.run_gsm8k(n=args.gsm8k, method=args.gsm8k_method)
        if args.math > 0:
            orchestrator.run_math(n=args.math)
        if args.humaneval > 0:
            orchestrator.run_humaneval(n=args.humaneval)
        if args.ablation:
            from cpptai.datasets import DatasetLoader
            problems = DatasetLoader.load_gsm8k(n=20)
            orchestrator.run_ablation_study(problems)

    orchestrator.generate_reports()
    print(f"\nDone. Reports in: {args.output_dir}")


if __name__ == "__main__":
    main()
