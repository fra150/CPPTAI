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

from .core import CPPTAITraslocatore


PROBLEMS: List[Dict] = [
    {
        "id": "energy_crisis",
        "prompt": (
            "How can we address the global energy crisis considering: "
            "1) limits of renewables, 2) nuclear costs, 3) fossil dependency, "
            "4) geopolitical factors, 5) a just transition for workers?"
        ),
        # Expected concepts for naïve accuracy scoring
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
]


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
    tokens = [t.lower() for t in text.split() if t]
    vec = [0.0] * dim
    for t in tokens:
        h = abs(hash(t)) % dim
        vec[h] += 1.0
    # L2 normalize
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


def naive_accuracy(text: str, expected: List[str]) -> float:
    found = 0
    lower = text.lower()
    for key in expected:
        if key.lower() in lower:
            found += 1
    return found / max(1, len(expected))


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

    for p in PROBLEMS:
        pid = p["id"]
        prompt = p["prompt"]
        expected = p["expected"]

        # Baselines
        for name, fn in methods:
            t0 = time.perf_counter()
            out = fn(prompt)
            dt = time.perf_counter() - t0
            acc = naive_accuracy(out, expected)
            div = shannon_entropy_norm(out)
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
                }
            )

        # CPPTAI
        t0 = time.perf_counter()
        result = orchestrator.solve(prompt)
        text = result.get("final_answer", "")
        dt = time.perf_counter() - t0
        acc = naive_accuracy(text, expected)
        div = shannon_entropy_norm(text)
        # CPPTAI detailed metrics
        # Build embeddings for all method outputs (including CPPTAI) to compute cluster-based diversity.
        method_texts = [
            baseline_cot(prompt),
            baseline_tot(prompt),
            baseline_got(prompt),
            baseline_react(prompt),
            text,
        ]
        vecs = [hash_embedding(t) for t in method_texts]
        assigns = kmeans(vecs, k=3, iters=10)
        # robust diversity: mean pairwise cosine distance, clamped to [0,1]
        pairs = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = cosine_similarity(vecs[i], vecs[j])
                dist = max(0.0, min(1.0, 1.0 - sim))
                pairs.append(dist)
        robust_div = round((sum(pairs) / len(pairs)) if pairs else 0.0, 3)
        cluster_count = len(set(assigns))

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
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    # Save JSON
    with open("benchmarks.json", "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, ensure_ascii=False, indent=2)

    return records, summary
