"""Massive GSM8K benchmark — runs all 1319 problems and generates comprehensive report."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from cpptai.types import RunConfig
from cpptai.datasets import DatasetLoader


REPORT_HEADERS = [
    "id", "problem", "expected_answer", "predicted_answer", "correct",
    "time_sec", "tokens_used", "trajectories_generated",
    "avg_confidence", "category", "error",
]

CATEGORY_KEYWORDS = {
    "1-step": ["sum", "total", "add", "plus", "+", "altogether"],
    "multi-step": ["first", "then", "after", "next", "finally", "each", "per"],
    "word_problem": ["if", "when", "how many", "how much", "what is", "find"],
}


def classify_problem(problem: str) -> str:
    pl = problem.lower()
    scores: Dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in pl)
    if not scores:
        return "other"
    return max(scores, key=scores.get)


def solve_single(problem_dict: Dict[str, Any], config: RunConfig, use_explorer: bool) -> Dict[str, Any]:
    problem = problem_dict["prompt"]
    expected_raw = problem_dict["expected"]
    expected = (expected_raw[0] if isinstance(expected_raw, list) else expected_raw).strip()
    category = classify_problem(problem)

    t0 = time.perf_counter()
    error = None
    predicted = ""
    tokens_used = 0
    trajectories_generated = 0
    avg_confidence = 0.0

    try:
        if use_explorer:
            from cpptai.exploration import ExplorerEngine, TrajectoryAnalyzer

            engine = ExplorerEngine(
                num_trajectories=config.explorer_num_trajectories,
                noise_level=config.explorer_noise_level,
                temperature=config.explorer_temperature,
                denoising_steps=config.explorer_denoising_steps,
                seed=config.seed + hash(problem) % (2**31),
                adaptive=config.explorer_adaptive,
                max_workers=1,
                parallel_llm=False,
                cache_results=config.explorer_cache_results,
                diversity_weight=config.explorer_diversity_weight,
            )
            trajectories = engine.explore(problem)
            trajectories_generated = len(trajectories)
            avg_confidence = sum(t.confidence for t in trajectories) / max(1, len(trajectories))

            analyzer = TrajectoryAnalyzer(
                ensemble_size=config.analyzer_ensemble_size,
                coherence_threshold=config.analyzer_coherence_threshold,
            )
            prepared = analyzer.analyze(trajectories, problem)
            enriched = prepared.enriched_problem

            from cpptai.pipeline_v2 import solve_with_cpptai
            result = solve_with_cpptai(enriched, config)
        else:
            from cpptai.pipeline_v2 import solve_with_cpptai
            result = solve_with_cpptai(problem, config)

        predicted = (result.get("answer") or result.get("final_answer") or "").strip()
        token_counts = result.get("token_counts", {})
        tokens_used = int(token_counts.get("final_answer_tokens", 0) if isinstance(token_counts, dict) else 0)
    except Exception as e:
        error = str(e)

    dt = time.perf_counter() - t0

    correct = False
    if predicted and expected:
        clean_pred = predicted.replace(",", "").replace("$", "").replace("%", "").strip()
        clean_exp = expected.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            correct = abs(float(clean_pred) - float(clean_exp)) < 0.01
        except (ValueError, TypeError):
            correct = clean_pred.lower() == clean_exp.lower()

    return {
        "id": problem_dict["id"],
        "problem": problem[:200],
        "expected_answer": expected,
        "predicted_answer": predicted,
        "correct": correct,
        "time_sec": round(dt, 3),
        "tokens_used": tokens_used,
        "trajectories_generated": trajectories_generated,
        "avg_confidence": round(avg_confidence, 4),
        "category": category,
        "error": error or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Massive GSM8K benchmark runner")
    parser.add_argument("--max-problems", type=int, default=1319, help="Max GSM8K problems to run")
    parser.add_argument("--output-dir", default="benchmarks/gsm8k_full")
    parser.add_argument("--explorer", action="store_true", help="Enable Explorer+Analyzer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial results")
    args = parser.parse_args()

    loader = DatasetLoader()
    problems = loader.load_gsm8k(n=args.max_problems)
    print(f"Loaded {len(problems)} GSM8K problems")

    os.makedirs(args.output_dir, exist_ok=True)

    config = RunConfig(
        seed=args.seed,
        explorer_enabled=args.explorer,
        analyzer_enabled=args.explorer,
        explorer_num_trajectories=8,
        explorer_noise_level=0.6,
        explorer_temperature=0.8,
        explorer_denoising_steps=3,
        explorer_adaptive=True,
        explorer_cache_results=True,
        explorer_diversity_weight=0.6,
        analyzer_ensemble_size=3,
        analyzer_coherence_threshold=0.5,
    )

    csv_path = os.path.join(args.output_dir, "results.csv")
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")

    completed_ids: set = set()
    results: List[Dict[str, Any]] = []

    if args.resume:
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    completed_ids.add(row["id"])
            print(f"Resumed {len(results)} completed results from {csv_path}")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
                completed_ids.update(cp.get("completed_ids", []))
            print(f"Resumed checkpoint with {len(completed_ids)} completed IDs")

    remaining = [p for p in problems if p["id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} problems to process")

    summary: Dict[str, Any] = {
        "total": len(problems),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "total_time": 0.0,
        "total_tokens": 0,
        "categories": {},
        "explorer_enabled": args.explorer,
    }

    batch_size = args.batch
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining) - 1) // batch_size + 1
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} problems) — {'Explorer ON' if args.explorer else 'Direct'}")

        batch_results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(solve_single, problem, config, args.explorer): idx
                for idx, problem in enumerate(batch)
            }
            for future in tqdm(as_completed(futures), total=len(batch), desc="  Solving"):
                idx = futures[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    batch_results[idx] = {
                        "id": batch[idx]["id"],
                        "problem": batch[idx]["prompt"][:200],
                        "expected_answer": batch[idx]["expected"][0] if isinstance(batch[idx]["expected"], list) else str(batch[idx]["expected"]),
                        "predicted_answer": "",
                        "correct": False,
                        "time_sec": 0.0,
                        "tokens_used": 0,
                        "trajectories_generated": 0,
                        "avg_confidence": 0.0,
                        "category": classify_problem(batch[idx]["prompt"]),
                        "error": str(e),
                    }

        for res in batch_results:
            if res is not None:
                results.append(res)
                if res["correct"]:
                    summary["correct"] += 1
                else:
                    if res["error"]:
                        summary["errors"] += 1
                    else:
                        summary["incorrect"] += 1
                summary["total_time"] += res["time_sec"]
                summary["total_tokens"] += res["tokens_used"]

                cat = res["category"]
                if cat not in summary["categories"]:
                    summary["categories"][cat] = {"total": 0, "correct": 0}
                summary["categories"][cat]["total"] += 1
                if res["correct"]:
                    summary["categories"][cat]["correct"] += 1

        write_results_csv(csv_path, results)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"completed_ids": [r["id"] for r in results]}, f)

        batch_accuracy = sum(1 for r in batch_results if r and r["correct"]) / max(1, len(batch_results))
        print(f"  Batch accuracy: {batch_accuracy:.2%}  |  Running accuracy: {summary['correct'] / max(1, summary['total']):.2%}")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Total problems:   {summary['total']}")
    print(f"Correct:          {summary['correct']} ({summary['correct'] / max(1, summary['total']):.2%})")
    print(f"Incorrect:        {summary['incorrect']}")
    print(f"Errors:           {summary['errors']}")
    print(f"Total time:       {summary['total_time']:.1f}s")
    print(f"Avg time/problem: {summary['total_time'] / max(1, summary['total']):.3f}s")
    print(f"Total tokens:     {summary['total_tokens']}")
    print(f"Explorer mode:    {'ON' if args.explorer else 'OFF'}")
    print()

    print("--- Accuracy by category ---")
    for cat, data in sorted(summary["categories"].items()):
        acc = data["correct"] / max(1, data["total"])
        print(f"  {cat:20s}: {data['correct']:4d}/{data['total']:<4d} ({acc:.2%})")

    write_summary_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_accuracy_chart(os.path.join(args.output_dir, "accuracy_chart.csv"), results)

    print(f"\nResults saved to {args.output_dir}/")


def write_results_csv(path: str, results: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    summary["accuracy"] = summary["correct"] / max(1, summary["total"])
    for cat in summary["categories"]:
        d = summary["categories"][cat]
        d["accuracy"] = d["correct"] / max(1, d["total"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_accuracy_chart(path: str, results: List[Dict[str, Any]]) -> None:
    correct_so_far = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_index", "cumulative_correct", "cumulative_accuracy"])
        for i, r in enumerate(results, 1):
            if r["correct"]:
                correct_so_far += 1
            writer.writerow([i, correct_so_far, round(correct_so_far / i, 6)])


if __name__ == "__main__":
    main()
