"""GARA CoT vs CPPTAI — 50 test di programmazione (HumanEval reali).

Stesse regole per entrambi: stesso prompt, stesso verificatore (esecuzione reale
dei test HumanEval in subprocess sandboxato), pass@1.

  CoT   : 1 chiamata DeepSeek (ragiona step-by-step, poi blocco ```python).
  CPPTAI: pipeline completa (discesa dual-track) + estrazione blocco ```python.

Uso: python scripts/run_coding_race.py --n 50 --workers 2
Output: benchmarks/coding_race/race_<timestamp>.csv + summary_<timestamp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from cpptai.datasets import DatasetLoader
from cpptai.deepseek_client import deepseek_chat, extract_text_answer
from cpptai.core import CPPTAITraslocatore
from cpptai.humaneval_executor import verify_humaneval, is_valid_python

CODING_TASK = (
    "Complete the Python function below. Think step by step, then output the COMPLETE "
    "function code inside one ```python block (code only inside the block, no ellipsis)."
)


def extract_code(text: str) -> str:
    """Estrae il codice dal blocco ```python, con fallback progressivi."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m and "def " in m.group(1):
        return m.group(1).strip()
    lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    if any(l.startswith("def ") or l.startswith("    ") or l.startswith("\t") for l in lines):
        return "\n".join(lines).strip()
    return text.strip()


def solve_cot(prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = deepseek_chat(
        [{"role": "system", "content": "You are an expert Python programmer."},
         {"role": "user", "content": f"{CODING_TASK}\n\n{prompt}"}],
        temperature=0,
        max_tokens=1024,
    )
    text = extract_text_answer(resp) if resp else ""
    return extract_code(text or ""), round(time.perf_counter() - t0, 2)


def solve_cpptai(prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = CPPTAITraslocatore(enable_phase_iv=False).solve(
        f"{CODING_TASK}\n\n{prompt}"
    )
    return extract_code(result.get("final_answer", "")), round(time.perf_counter() - t0, 2)


def race_one(problem: dict) -> dict:
    test_code = problem.get("test", "")
    entry = problem.get("entry_point", "")
    rec: dict = {"problem_id": problem["id"], "task_id": problem.get("task_id", problem["id"])}

    cot_code, cot_t = solve_cot(problem["prompt"])
    cpptai_code, cpptai_t = solve_cpptai(problem["prompt"])

    for prefix, code, t in (("cot", cot_code, cot_t), ("cpptai", cpptai_code, cpptai_t)):
        ok, _ = is_valid_python(code)
        rec[f"{prefix}_syntax"] = int(ok)
        rec[f"{prefix}_time"] = t
        if ok and test_code and entry:
            res = verify_humaneval(code, test_code, entry)
            rec[f"{prefix}_pass"] = int(bool(res["passed"]))
        else:
            rec[f"{prefix}_pass"] = 0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Gara CoT vs CPPTAI su HumanEval")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    print(f"\n=== GARA CoT vs CPPTAI — {args.n} problemi HumanEval (workers={args.workers}) ===\n")
    problems = DatasetLoader().load_humaneval(n=args.n)
    real = sum(1 for p in problems if p.get("test"))
    print(f"Caricati {len(problems)} problemi ({real} con test reali).\n")

    records: list = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(race_one, p): p for p in problems}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"problem_id": p["id"], "task_id": p.get("task_id", p["id"]),
                     "cot_syntax": 0, "cot_time": 0.0, "cot_pass": 0,
                     "cpptai_syntax": 0, "cpptai_time": 0.0, "cpptai_pass": 0,
                     "error": str(e)[:200]}
            records.append(r)
            c = sum(x["cot_pass"] for x in records)
            k = sum(x["cpptai_pass"] for x in records)
            print(f"  {i}/{len(problems)} | CoT {c} - {k} CPPTAI | {r['problem_id']} "
                  f"cot={'PASS' if r['cot_pass'] else 'fail'}({r['cot_time']}s) "
                  f"cpptai={'PASS' if r['cpptai_pass'] else 'fail'}({r['cpptai_time']}s)", flush=True)

    n = max(1, len(records))
    summary = {
        "n": len(records),
        "cot_pass": sum(x["cot_pass"] for x in records),
        "cpptai_pass": sum(x["cpptai_pass"] for x in records),
        "cot_acc": round(sum(x["cot_pass"] for x in records) / n, 3),
        "cpptai_acc": round(sum(x["cpptai_pass"] for x in records) / n, 3),
        "cot_syntax_ok": sum(x["cot_syntax"] for x in records),
        "cpptai_syntax_ok": sum(x["cpptai_syntax"] for x in records),
        "cot_tempo_medio": round(sum(x["cot_time"] for x in records) / n, 1),
        "cpptai_tempo_medio": round(sum(x["cpptai_time"] for x in records) / n, 1),
        "tempo_totale_s": round(time.perf_counter() - t0, 1),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("benchmarks/coding_race")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"race_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["problem_id", "task_id", "cot_syntax", "cot_pass",
                                          "cot_time", "cpptai_syntax", "cpptai_pass", "cpptai_time"])
        w.writeheader()
        for r in sorted(records, key=lambda x: x["problem_id"]):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    json_path = outdir / f"summary_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== RISULTATO ({summary['tempo_totale_s']}s) ===")
    print(f"CoT    {summary['cot_pass']}/{summary['n']} = {summary['cot_acc']} "
          f"(sintassi ok {summary['cot_syntax_ok']}, {summary['cot_tempo_medio']}s/problema)")
    print(f"CPPTAI {summary['cpptai_pass']}/{summary['n']} = {summary['cpptai_acc']} "
          f"(sintassi ok {summary['cpptai_syntax_ok']}, {summary['cpptai_tempo_medio']}s/problema)")
    print(f"CSV: {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
