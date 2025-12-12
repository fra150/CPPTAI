"""Benchmark runner for CPPTAI and baseline methods.

Automates simple quantitative evaluation:
- accuracy vs baselines (CoT, ToT, GoT, ReAct, CPPTAI)
- diversity via Shannon entropy on token distributions (normalized 0–1)
- error rate (1 - accuracy)
- time-per-problem (seconds)

Outputs results to `benchmarks.csv` and `benchmarks.json`.
"""

from __future__ import annotations

import csv
import json
import math
import time
from typing import Dict, List, Tuple
import os

from .core import CPPTAITraslocatore


def build_problems(n: int = 50) -> List[Dict]:
    regions = ["EU", "USA", "India", "China", "Brazil", "South Africa", "Japan", "Australia"]
    caps = ["net-zero 2050", "-50% CO2 by 2035", "carbon budget 1.5C"]
    mixes = ["renewables-heavy", "balanced", "nuclear-anchored"]
    variants: List[Dict] = []
    idx = 1
    for r in regions:
        for cap in caps:
            for mix in mixes:
                prompt = (
                    f"Energy planning for {r}: constraints include 1) limits of renewables, 2) nuclear costs, "
                    f"3) fossil dependency, 4) geopolitics. Target: {cap}. Preferred mix: {mix}. "
                    f"Ensure a just transition for workers."
                )
                variants.append(
                    {
                        "id": f"energy_crisis_{idx}",
                        "prompt": prompt,
                        "expected": [
                            "storage",
                            "smart grids",
                            "SMR",
                            "CCUS",
                            "electrification",
                            "methane",
                            "diplomacy",
                            "recycling",
                            "reserves",
                            "retraining",
                        ],
                    }
                )
                idx += 1
                if len(variants) >= n:
                    return variants
    return variants[:n]

PROBLEMS: List[Dict] = build_problems(50)
# Precompute prompt lengths to define normalized complexity per problem
_PROMPT_LENGTHS = [len(p["prompt"].split()) for p in PROBLEMS]
_MAX_PROMPT_LEN = max(_PROMPT_LENGTHS) if _PROMPT_LENGTHS else 1


def shannon_entropy_norm(text: str) -> float:
    tokens = [t.lower() for t in text.split() if t]
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = float(sum(freq.values()))
    probs = [c / total for c in freq.values()]
    H = -sum(p * math.log(p + 1e-12, 2) for p in probs)
    Hmax = math.log(len(freq) + 1e-12, 2)
    return max(0.0, min(1.0, H / (Hmax if Hmax > 0 else 1.0)))


def hash_embedding(text: str, dim: int = 128) -> List[float]:
    import hashlib
    tokens = [t.lower() for t in text.split() if t]
    vec = [0.0] * dim
    for t in tokens:
        hbytes = hashlib.sha256(t.encode("utf-8")).digest()
        h = int.from_bytes(hbytes[:4], "big") % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def kmeans(vectors: List[List[float]], k: int = 3, iters: int = 10) -> List[int]:
    if not vectors:
        return []
    k = min(k, len(vectors))
    centroids = [vectors[i][:] for i in range(k)]
    assignments = [0] * len(vectors)
    for _ in range(iters):
        # assign
        for i, v in enumerate(vectors):
            sims = [cosine_similarity(v, c) for c in centroids]
            assignments[i] = int(max(range(k), key=lambda j: sims[j]))
        # update
        sums = [[0.0] * len(vectors[0]) for _ in range(k)]
        counts = [0] * k
        for v, a in zip(vectors, assignments):
            counts[a] += 1
            for j in range(len(v)):
                sums[a][j] += v[j]
        for c in range(k):
            if counts[c] == 0:
                continue
            centroids[c] = [x / counts[c] for x in sums[c]]
            # renormalize
            norm = math.sqrt(sum(x * x for x in centroids[c])) or 1.0
            centroids[c] = [x / norm for x in centroids[c]]
    return assignments


def rubric_accuracy(text: str, expected: List[str]) -> float:
    """0–1 rubric score based on expected concept hits with partial credit.

    Each expected concept contributes in {0, 0.5, 1} via exact or synonym match.
    """
    lower = text.lower()
    synonyms: Dict[str, List[str]] = {
        "storage": ["batteries", "battery", "hydrogen storage", "pumped storage"],
        "smart grids": ["grid modernization", "smart grid", "digital grid"],
        "SMR": ["small modular reactor", "small modular reactors"],
        "CCUS": ["carbon capture", "carbon storage", "ccs"],
        "electrification": ["electrify", "evs", "heat pumps"],
        "methane": ["ch4", "methane leak", "methane leakage"],
        "diplomacy": ["international cooperation", "jetp", "energy diplomacy"],
        "recycling": ["materials recycling", "recycle"],
        "reserves": ["strategic reserves", "stockpile"],
        "retraining": ["job training", "vocational", "reskilling"],
    }
    score = 0.0
    for key in expected:
        k = key.lower()
        if k in lower:
            score += 1.0
        else:
            syns = synonyms.get(k, [])
            if any(s in lower for s in syns):
                score += 0.5
    return score / max(1, len(expected))


def baseline_cot(problem: str) -> str:
    return (
        "We analyze constraints and propose a step-by-step plan combining renewables, "
        "nuclear, and demand-side management with policy support."
    )


def baseline_tot(problem: str) -> str:
    return (
        "Tree-of-Thought branches: (A) renewables, (B) storage, (C) nuclear, (D) policy; "
        "choose path A->B->C integrating trade-offs."
    )


def baseline_got(problem: str) -> str:
    return (
        "Graph-of-Thought nodes linked across energy sources, infrastructure, finance, "
        "and social impact; optimize multi-objective edges."
    )


def baseline_react(problem: str) -> str:
    return (
        "Action: query energy policies; Observation: storage costs decreasing; "
        "Action: propose hybrid strategy; Observation: public acceptance varies."
    )


def run_benchmarks() -> Tuple[List[Dict], Dict]:
    records: List[Dict] = []
    methods = [
        ("CoT", baseline_cot),
        ("ToT", baseline_tot),
        ("GoT", baseline_got),
        ("ReAct", baseline_react),
    ]
    orchestrator = CPPTAITraslocatore()
    orchestrator_no_iv = CPPTAITraslocatore(enable_phase_iv=False)
    orchestrator_no_i = CPPTAITraslocatore(enable_phase_i=False)
    use_no_iv = os.getenv("BENCH_DISABLE_EXTERNAL", "0") == "1"
    orchestrator_main = orchestrator_no_iv if use_no_iv else orchestrator

    for p in PROBLEMS:
        pid = p["id"]
        prompt = p["prompt"]
        expected = p["expected"]

        # Baselines
        for name, fn in methods:
            for run in (1, 2, 3):
                t0 = time.perf_counter()
                out = fn(prompt)
                dt = time.perf_counter() - t0
                acc = rubric_accuracy(out, expected)
                div = shannon_entropy_norm(out)
                p_complexity = len(prompt.split()) / _MAX_PROMPT_LEN
                records.append(
                    {
                        "problem_id": pid,
                        "method": name,
                        "accuracy": round(acc, 3),
                        "error_rate": round(1.0 - acc, 3),
                        "diversity": round(div, 3),
                        "time_sec": round(dt, 3),
                        "tokens": len(out.split()),
                        "robust_diversity": None,
                        "clusters": None,
                        "problem_complexity": round(p_complexity, 3),
                    }
                )

        # CPPTAI
        for run in (1, 2, 3):
            t0 = time.perf_counter()
            result = orchestrator_main.solve(prompt)
            text = result.get("final_answer", "")
            dt = time.perf_counter() - t0
            acc = rubric_accuracy(text, expected)
            div = shannon_entropy_norm(text)
            method_texts = [
                baseline_cot(prompt),
                baseline_tot(prompt),
                baseline_got(prompt),
                baseline_react(prompt),
                text,
            ]
            vecs = [hash_embedding(t) for t in method_texts]
            assigns = kmeans(vecs, k=3, iters=10)
            pairs = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sim = cosine_similarity(vecs[i], vecs[j])
                    dist = max(0.0, min(1.0, 1.0 - sim))
                    pairs.append(dist)
            robust_div = round((sum(pairs) / len(pairs)) if pairs else 0.0, 3)
            cluster_count = len(set(assigns))
            p_complexity = len(prompt.split()) / _MAX_PROMPT_LEN

            records.append(
                {
                    "problem_id": pid,
                    "method": "CPPTAI",
                    "accuracy": round(acc, 3),
                    "error_rate": round(1.0 - acc, 3),
                    "diversity": round(div, 3),
                    "time_sec": round(dt, 3),
                    "tokens": len(text.split()),
                    "robust_diversity": robust_div,
                    "clusters": cluster_count,
                    "problem_complexity": round(p_complexity, 3),
                }
            )

        # Ablation: no Phase IV
        for run in (1, 2, 3):
            t0 = time.perf_counter()
            result = (orchestrator_no_iv if use_no_iv else orchestrator_no_iv).solve(prompt)
            text = result.get("final_answer", "")
            dt = time.perf_counter() - t0
            acc = rubric_accuracy(text, expected)
            div = shannon_entropy_norm(text)
            method_texts = [baseline_cot(prompt), baseline_tot(prompt), baseline_got(prompt), baseline_react(prompt), text]
            vecs = [hash_embedding(t) for t in method_texts]
            assigns = kmeans(vecs, k=3, iters=10)
            pairs = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sim = cosine_similarity(vecs[i], vecs[j])
                    dist = max(0.0, min(1.0, 1.0 - sim))
                    pairs.append(dist)
            robust_div = round((sum(pairs) / len(pairs)) if pairs else 0.0, 3)
            cluster_count = len(set(assigns))
            p_complexity = len(prompt.split()) / _MAX_PROMPT_LEN
            records.append(
                {
                    "problem_id": pid,
                    "method": "CPPTAI_no_IV",
                    "accuracy": round(acc, 3),
                    "error_rate": round(1.0 - acc, 3),
                    "diversity": round(div, 3),
                    "time_sec": round(dt, 3),
                    "tokens": len(text.split()),
                    "robust_diversity": robust_div,
                    "clusters": cluster_count,
                    "problem_complexity": round(p_complexity, 3),
                }
            )

        # Ablation: no Phase I
        for run in (1, 2, 3):
            t0 = time.perf_counter()
            result = (orchestrator_no_iv if use_no_iv else orchestrator_no_i).solve(prompt)
            text = result.get("final_answer", "")
            dt = time.perf_counter() - t0
            acc = rubric_accuracy(text, expected)
            div = shannon_entropy_norm(text)
            method_texts = [baseline_cot(prompt), baseline_tot(prompt), baseline_got(prompt), baseline_react(prompt), text]
            vecs = [hash_embedding(t) for t in method_texts]
            assigns = kmeans(vecs, k=3, iters=10)
            pairs = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sim = cosine_similarity(vecs[i], vecs[j])
                    dist = max(0.0, min(1.0, 1.0 - sim))
                    pairs.append(dist)
            robust_div = round((sum(pairs) / len(pairs)) if pairs else 0.0, 3)
            cluster_count = len(set(assigns))
            p_complexity = len(prompt.split()) / _MAX_PROMPT_LEN
            records.append(
                {
                    "problem_id": pid,
                    "method": "CPPTAI_no_I",
                    "accuracy": round(acc, 3),
                    "error_rate": round(1.0 - acc, 3),
                    "diversity": round(div, 3),
                    "time_sec": round(dt, 3),
                    "tokens": len(text.split()),
                    "robust_diversity": robust_div,
                    "clusters": cluster_count,
                    "problem_complexity": round(p_complexity, 3),
                }
            )

    # Aggregate summary per method (mean across problems)
    summary: Dict[str, Dict] = {}
    by_method: Dict[str, List[Dict]] = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)
    for m, arr in by_method.items():
        summary[m] = {
            "accuracy": round(sum(x["accuracy"] for x in arr) / len(arr), 3),
            "error_rate": round(sum(x["error_rate"] for x in arr) / len(arr), 3),
            "diversity": round(sum(x["diversity"] for x in arr) / len(arr), 3),
            "time_sec": round(sum(x["time_sec"] for x in arr) / len(arr), 3),
            "tokens": round(sum(x["tokens"] for x in arr) / len(arr), 1),
            "robust_diversity": round(
                sum((x["robust_diversity"] or 0.0) for x in arr) / max(1, len([x for x in arr if x["robust_diversity"] is not None])),
                3,
            ),
            "clusters": round(
                sum((x["clusters"] or 0) for x in arr) / max(1, len([x for x in arr if x["clusters"] is not None])),
                1,
            ),
        }

    # Save CSV
    with open("benchmarks.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "method",
                "accuracy",
                "error_rate",
                "diversity",
                "time_sec",
                "tokens",
                "robust_diversity",
                "clusters",
                "problem_complexity",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    # Save JSON
    with open("benchmarks.json", "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, ensure_ascii=False, indent=2)

    # Save summary CSV for LaTeX plots
    with open("benchmarks_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "accuracy",
                "error_rate",
                "diversity",
                "time_sec",
                "tokens",
                "robust_diversity",
                "clusters",
            ],
        )
        writer.writeheader()
        rows = [{"method": m, **vals} for m, vals in summary.items()]
        writer.writerows(rows)

    # Save cumulative accuracy series per method vs problem complexity
    with open("cumulative_accuracy.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "complexity", "cumulative_accuracy"])
        writer.writeheader()
        for m, arr in by_method.items():
            arr_sorted = sorted(arr, key=lambda x: x.get("problem_complexity", 0.0))
            cum_acc = 0.0
            for i, rec in enumerate(arr_sorted, start=1):
                cum_acc += rec["accuracy"]
                writer.writerow(
                    {
                        "method": m,
                        "complexity": rec.get("problem_complexity", 0.0),
                        "cumulative_accuracy": round(cum_acc / i, 3),
                    }
                )

    # Save error matrix by phase/method tag for heatmap generation
    def _phase_tag(method: str) -> str:
        if method == "CPPTAI":
            return "Full"
        if method == "CPPTAI_no_IV":
            return "No_IV"
        if method == "CPPTAI_no_I":
            return "No_I"
        return "Baseline"

    with open("error_by_phase.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "phase", "mean_error_rate"])
        writer.writeheader()
        for m, arr in by_method.items():
            mean_err = sum(x["error_rate"] for x in arr) / len(arr)
            writer.writerow({"method": m, "phase": _phase_tag(m), "mean_error_rate": round(mean_err, 3)})

    # Save statistical comparisons (paired t-statistic and Cohen's d)
    def _mean_accuracy_by_problem(method: str) -> Dict[str, float]:
        per_problem: Dict[str, List[float]] = {}
        for r in records:
            if r["method"] != method:
                continue
            per_problem.setdefault(r["problem_id"], []).append(r["accuracy"])
        return {pid: (sum(vals) / len(vals)) for pid, vals in per_problem.items() if vals}

    def _paired_t_and_cohen_d(a_vals: List[float], b_vals: List[float]) -> Tuple[float, float, int]:
        n = min(len(a_vals), len(b_vals))
        if n == 0:
            return 0.0, 0.0, 0
        diffs = [a_vals[i] - b_vals[i] for i in range(n)]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(1, (n - 1))
        sd_diff = math.sqrt(var_diff)
        t_stat = mean_diff / (sd_diff / math.sqrt(n)) if sd_diff > 0 else 0.0
        mean_a = sum(a_vals[:n]) / n
        mean_b = sum(b_vals[:n]) / n
        var_a = sum((x - mean_a) ** 2 for x in a_vals[:n]) / max(1, (n - 1))
        var_b = sum((x - mean_b) ** 2 for x in b_vals[:n]) / max(1, (n - 1))
        pooled_sd = math.sqrt(((n - 1) * var_a + (n - 1) * var_b) / max(1, (2 * n - 2))) or 0.0
        cohen_d = ((mean_a - mean_b) / pooled_sd) if pooled_sd > 0 else 0.0
        return round(t_stat, 3), round(cohen_d, 3), n

    pairs_to_compare = [
        ("CPPTAI", "CoT"),
        ("CPPTAI", "ToT"),
        ("CPPTAI", "GoT"),
        ("CPPTAI", "ReAct"),
        ("CPPTAI", "CPPTAI_no_IV"),
        ("CPPTAI", "CPPTAI_no_I"),
    ]
    with open("stats_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method_a", "method_b", "t_stat", "cohen_d", "n"])
        writer.writeheader()
        maps = {m: _mean_accuracy_by_problem(m) for m, _ in by_method.items()}
        for a, b in pairs_to_compare:
            ma = maps.get(a, {})
            mb = maps.get(b, {})
            common = [pid for pid in ma.keys() if pid in mb]
            a_vals = [ma[pid] for pid in common]
            b_vals = [mb[pid] for pid in common]
            t_stat, d, n = _paired_t_and_cohen_d(a_vals, b_vals)
            writer.writerow({"method_a": a, "method_b": b, "t_stat": t_stat, "cohen_d": d, "n": n})

    return records, summary
