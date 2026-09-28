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
from typing import Dict, List, Optional, Tuple
import os
import re

from .core import CPPTAITraslocatore
from .datasets import DatasetLoader, get_all_datasets
from .baselines import CoTBaseline, ToTBaseline, GoTBaseline, ReActBaseline


def compute_pass_at_k(records: List[Dict], k: int = 1) -> float:
    """Compute pass@k from benchmark records."""
    correct = sum(1 for r in records if r.get("accuracy", 0) >= 1.0)
    return round(correct / max(1, len(records)), 4)


def category_breakdown(records: List[Dict]) -> Dict:
    """Break down accuracy by problem category."""
    categories = {}
    for r in records:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.get("accuracy", 0) >= 1.0:
            categories[cat]["correct"] += 1
    for cat in categories:
        c = categories[cat]
        c["accuracy"] = round(c["correct"] / max(1, c["total"]), 3)
    return categories


def build_problems(n: int = 50) -> List[Dict]:
    # Legacy energy problems
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
                        "dataset": "energy_synthetic"
                    }
                )
                idx += 1
                if len(variants) >= n:
                    break
            if len(variants) >= n:
                break
    
    # Mix in broader benchmark suite (stubs for GSM8K, MATH, etc.)
    variants.extend(get_all_datasets(n_per_set=5))
    return variants

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
    Supports basic numeric tolerance for math problems.
    """
    lower = text.lower()
    
    # Domain-specific synonyms for the energy problem
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
        # Direct match
        if k in lower:
            score += 1.0
            continue
            
        # Synonym match
        syns = synonyms.get(k, [])
        if any(s in lower for s in syns):
            score += 0.5
            continue
            
        # Numeric match (simple heuristic)
        if k.replace('.', '', 1).isdigit():
            import re
            # Extract all numbers from text
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", lower)
            try:
                target = float(k)
                # Check if any number in text is close to target (within 1%)
                if any(abs(float(n) - target) < max(0.01, 0.01 * abs(target)) for n in nums):
                    score += 1.0
            except ValueError:
                pass
                
    return score / max(1, len(expected))


def _extract_final_number_str(text: str) -> Optional[str]:
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").strip()
    if raw.endswith("."):
        raw = raw[:-1]
    return raw if raw else None


def gsm8k_accuracy(text: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    pred_s = _extract_final_number_str(text) or text.strip()
    exp_s = _extract_final_number_str(expected[0]) or expected[0].strip()
    try:
        pred = float(pred_s)
        exp = float(exp_s)
    except (ValueError, TypeError):
        return 0.0
    return 1.0 if abs(pred - exp) <= 1e-9 else 0.0


def run_gsm8k_cpptai(n: int = 30) -> Tuple[List[Dict], Dict]:
    orchestrator = CPPTAITraslocatore(enable_phase_iv=False)
    items = DatasetLoader.load_gsm8k(n=n)
    records: List[Dict] = []
    correct = 0
    for item in items:
        pid = item["id"]
        prompt = item["prompt"]
        expected = item.get("expected") or []
        t0 = time.perf_counter()
        res = orchestrator.solve_gsm8k(prompt)
        dt = time.perf_counter() - t0
        pred = res.get("final_answer", "")
        acc = gsm8k_accuracy(pred, expected)
        correct += 1 if acc >= 1.0 else 0
        records.append(
            {
                "problem_id": pid,
                "method": "CPPTAI_gsm8k",
                "accuracy": round(acc, 3),
                "time_sec": round(dt, 3),
                "expected": expected[0] if expected else "",
                "prediction": pred,
            }
        )
    total = len(records) or 1
    summary = {"n": total, "accuracy": round(correct / total, 3)}
    return records, summary


# ---------------------------------------------------------------------------
# Helper: diversity metrics (hash, kmeans, cosine, robust_div)
# ---------------------------------------------------------------------------

def _compute_diversity_metrics(texts: List[str]) -> Dict:
    """Compute robust diversity and cluster count for a list of texts."""
    vecs = [hash_embedding(t) for t in texts]
    assigns = kmeans(vecs, k=3, iters=10)
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = cosine_similarity(vecs[i], vecs[j])
            pairs.append(max(0.0, min(1.0, 1.0 - sim)))
    return {
        "robust_diversity": round((sum(pairs) / len(pairs)) if pairs else 0.0, 3),
        "clusters": len(set(assigns)),
    }


def _make_record(prompt: str, expected: List[str], dataset: str, method: str,
                 text: str, dt: float, diversity: Dict) -> Dict:
    """Build a single benchmark record dict."""
    acc = gsm8k_accuracy(text, expected) if dataset == "gsm8k" else rubric_accuracy(text, expected)
    div = shannon_entropy_norm(text)
    return {
        "problem_id": "",
        "method": method,
        "accuracy": round(acc, 3),
        "error_rate": round(1.0 - acc, 3),
        "diversity": round(div, 3),
        "time_sec": round(dt, 3),
        "tokens": len(text.split()),
        "robust_diversity": diversity.get("robust_diversity"),
        "clusters": diversity.get("clusters"),
        "problem_complexity": 0.0,
    }


def _solve_cpptai(orchestrator, prompt: str, dataset: str) -> str:
    """Run CPPTAI solver and return the answer text."""
    if dataset == "gsm8k":
        result = orchestrator.solve_gsm8k(prompt)
    else:
        result = orchestrator.solve(prompt)
    return result.get("final_answer", "")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks() -> Tuple[List[Dict], Dict]:
    records: List[Dict] = []

    # Baseline functions (wrapped for lazy call)
    baselines = [
        ("CoT", lambda p: CoTBaseline().solve(p)),
        ("ToT", lambda p: ToTBaseline().solve(p)),
        ("GoT", lambda p: GoTBaseline().solve(p)),
        ("ReAct", lambda p: ReActBaseline().solve(p)),
    ]

    orchestrator = CPPTAITraslocatore()
    orchestrator_no_iv = CPPTAITraslocatore(enable_phase_iv=False)
    orchestrator_no_i = CPPTAITraslocatore(enable_phase_i=False)
    use_no_iv = os.getenv("BENCH_DISABLE_EXTERNAL", "0") == "1"
    orchestrator_main = orchestrator_no_iv if use_no_iv else orchestrator

    # Ablation configs: (method_name, orchestrator)
    ablation_configs = [
        ("CPPTAI", orchestrator_main),
        ("CPPTAI_no_IV", orchestrator_no_iv),
        ("CPPTAI_no_I", orchestrator_no_i),
    ]

    for p in PROBLEMS:
        pid = p["id"]
        prompt = p["prompt"]
        expected = p["expected"]
        dataset = p.get("dataset", "")
        p_complexity = len(prompt.split()) / _MAX_PROMPT_LEN

        # --- Baselines ---
        for name, fn in baselines:
            for _ in range(3):
                t0 = time.perf_counter()
                out = fn(prompt)
                dt = time.perf_counter() - t0
                rec = _make_record(prompt, expected, dataset, name, out, dt, {})
                rec["problem_id"] = pid
                rec["problem_complexity"] = round(p_complexity, 3)
                records.append(rec)

        # --- CPPTAI variants (loop over ablation configs) ---
        for method_name, orch in ablation_configs:
            for _ in range(3):
                t0 = time.perf_counter()
                text = _solve_cpptai(orch, prompt, dataset)
                dt = time.perf_counter() - t0
                # Compute diversity with baselines for context
                baseline_texts = [fn(prompt) for _, fn in baselines]
                diversity = _compute_diversity_metrics(baseline_texts + [text])
                rec = _make_record(prompt, expected, dataset, method_name, text, dt, diversity)
                rec["problem_id"] = pid
                rec["problem_complexity"] = round(p_complexity, 3)
                records.append(rec)

    # --- Aggregate summary per method ---
    summary, by_method = _aggregate_summary(records)
    _save_all_reports(records, by_method, summary)
    return records, summary


# ---------------------------------------------------------------------------
# Aggregation and reporting helpers
# ---------------------------------------------------------------------------

def _aggregate_summary(records: List[Dict]) -> Tuple[Dict, Dict[str, List[Dict]]]:
    """Aggregate records by method and compute mean metrics."""
    by_method: Dict[str, List[Dict]] = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)

    summary: Dict[str, Dict] = {}
    for m, arr in by_method.items():
        n = len(arr)
        summary[m] = {
            "accuracy": round(sum(x["accuracy"] for x in arr) / n, 3),
            "error_rate": round(sum(x["error_rate"] for x in arr) / n, 3),
            "diversity": round(sum(x["diversity"] for x in arr) / n, 3),
            "time_sec": round(sum(x["time_sec"] for x in arr) / n, 3),
            "tokens": round(sum(x["tokens"] for x in arr) / n, 1),
        }
        rd_vals = [x["robust_diversity"] for x in arr if x["robust_diversity"] is not None]
        if rd_vals:
            summary[m]["robust_diversity"] = round(sum(rd_vals) / len(rd_vals), 3)
        cl_vals = [x["clusters"] for x in arr if x["clusters"] is not None]
        if cl_vals:
            summary[m]["clusters"] = round(sum(cl_vals) / len(cl_vals), 1)
    return summary, by_method


RECORD_FIELDS = [
    "problem_id", "method", "accuracy", "error_rate", "diversity",
    "time_sec", "tokens", "robust_diversity", "clusters", "problem_complexity",
]


def _save_csv(path: str, records: List[Dict], fields: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def _phase_tag(method: str) -> str:
    tags = {"CPPTAI": "Full", "CPPTAI_no_IV": "No_IV", "CPPTAI_no_I": "No_I"}
    return tags.get(method, "Baseline")


def _mean_accuracy_by_problem(records: List[Dict], method: str) -> Dict[str, float]:
    per_problem: Dict[str, List[float]] = {}
    for r in records:
        if r["method"] == method:
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


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_value_from_t(t: float, n: int) -> float:
    z = abs(t)
    return round(max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(z)))), 6)


def _save_all_reports(records: List[Dict], by_method: Dict[str, List[Dict]], summary: Dict) -> None:
    """Write all CSV/JSON output files."""
    # 1. Full records CSV
    _save_csv("benchmarks.csv", records, RECORD_FIELDS)

    # 2. JSON
    with open("benchmarks.json", "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, ensure_ascii=False, indent=2)

    # 3. Summary CSV
    sf = ["method", "accuracy", "error_rate", "diversity", "time_sec", "tokens"]
    rows = [{"method": m, **{k: v for k, v in vals.items() if k in sf}} for m, vals in summary.items()]
    _save_csv("benchmarks_summary.csv", rows, sf)

    # 4. Cumulative accuracy by complexity
    with open("cumulative_accuracy.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "complexity", "cumulative_accuracy"])
        w.writeheader()
        for m, arr in by_method.items():
            arr_sorted = sorted(arr, key=lambda x: x.get("problem_complexity", 0.0))
            cum = 0.0
            for i, rec in enumerate(arr_sorted, start=1):
                cum += rec["accuracy"]
                w.writerow({"method": m, "complexity": rec.get("problem_complexity", 0.0),
                           "cumulative_accuracy": round(cum / i, 3)})

    # 5. Error by phase
    with open("error_by_phase.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "phase", "mean_error_rate"])
        w.writeheader()
        for m, arr in by_method.items():
            w.writerow({"method": m, "phase": _phase_tag(m),
                       "mean_error_rate": round(sum(x["error_rate"] for x in arr) / len(arr), 3)})

    # 6. Statistical comparisons
    pairs = [
        ("CPPTAI", "CoT"), ("CPPTAI", "ToT"), ("CPPTAI", "GoT"),
        ("CPPTAI", "ReAct"), ("CPPTAI", "CPPTAI_no_IV"), ("CPPTAI", "CPPTAI_no_I"),
    ]
    with open("stats_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method_a", "method_b", "t_stat", "cohen_d", "n", "p_value"])
        w.writeheader()
        maps = {m: _mean_accuracy_by_problem(records, m) for m in by_method}
        for a, b in pairs:
            ma, mb = maps.get(a, {}), maps.get(b, {})
            common = [pid for pid in ma if pid in mb]
            a_vals, b_vals = [ma[pid] for pid in common], [mb[pid] for pid in common]
            t_stat, d, n = _paired_t_and_cohen_d(a_vals, b_vals)
            w.writerow({"method_a": a, "method_b": b, "t_stat": t_stat,
                       "cohen_d": d, "n": n, "p_value": _p_value_from_t(t_stat, n)})
