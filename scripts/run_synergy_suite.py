"""Suite sinergia CoT+CPPTAI su N problemi GSM8K reali.

Per ogni problema: 5 step (pensiero -> analisi -> meta -> riduzione -> output
CoT+CPPTAI sul ridotto) + baseline CoT sull'originale per confronto.

Uso: python scripts/run_synergy_suite.py --n 20 --workers 2
Output: benchmarks/synergy/synergy_<timestamp>.csv + summary_<timestamp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from cpptai.datasets import DatasetLoader
from cpptai.deepseek_client import deepseek_chat, extract_text_answer
from cpptai.baselines import CoTBaseline
from cpptai.core import CPPTAITraslocatore
from cpptai.benchmarks import gsm8k_accuracy


def ask_ai(system: str, user: str, max_tokens: int = 512) -> str:
    resp = deepseek_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
    )
    text = extract_text_answer(resp) if resp else ""
    return (text or "").strip()


def synergy_one(problem: str, expected: list) -> dict:
    t0 = time.perf_counter()
    out: dict = {}

    # Step 1 — pensiero
    pensieri = ask_ai(
        "You are a careful thinker. Write down every raw thought and observation about the problem data.",
        f"Think aloud about this problem, listing all data, quantities and relations you notice:\n\n{problem}",
    )
    # Step 2 — analisi dati
    analisi = ask_ai(
        "You are a data analyst. Extract structured facts only, no chatter.",
        "From these raw thoughts, extract the structured data (entities, numbers, relations, question asked). "
        f"One fact per line starting with '-'.\n\nThoughts:\n{pensieri}",
    )
    # Step 3 — analisi del pensiero
    meta = ask_ai(
        "You are a meta-cognition auditor. Judge the thinking quality briefly and honestly.",
        f"Raw thoughts:\n{pensieri}\n\nData analysis:\n{analisi}\n\n"
        "Answer in 3 short lines: 1) is the thinking coherent? 2) is anything missing or wrong? 3) verdict: SOLID or WEAK.",
    )
    # Step 4 — riduzione
    ridotto = ask_ai(
        "You are a minimalist editor. Keep only what is needed to solve the problem. Reply with the reduced problem only.",
        f"Original problem:\n{problem}\n\nStructured data:\n{analisi}\n\nMeta verdict:\n{meta}\n\n"
        "Rewrite the problem keeping ONLY the data needed to solve it. Drop everything else.",
    )
    if not ridotto:
        ridotto = problem  # fallback: mai bloccare la pipeline

    # Step 5 — output insieme: CoT e CPPTAI sullo stesso ridotto + baseline CoT su originale
    cot_red = CoTBaseline().solve(ridotto)
    cot_orig = CoTBaseline().solve(problem)
    cpptai_answer = CPPTAITraslocatore(enable_phase_iv=False).solve(ridotto).get("final_answer", "")

    out["cot_orig_acc"] = gsm8k_accuracy(cot_orig, expected)
    out["cot_red_acc"] = gsm8k_accuracy(cot_red, expected)
    out["cpptai_red_acc"] = gsm8k_accuracy(cpptai_answer, expected)
    out["sinergia_ok"] = int(out["cot_red_acc"] == 1.0 or out["cpptai_red_acc"] == 1.0)
    out["orig_len"] = len(problem)
    out["red_len"] = len(ridotto)
    out["time_sec"] = round(time.perf_counter() - t0, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite sinergia CoT+CPPTAI su GSM8K")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    print(f"\n=== SINERGIA CoT+CPPTAI su {args.n} problemi GSM8K (workers={args.workers}) ===\n")
    problems = DatasetLoader().load_gsm8k(n=args.n)
    print(f"Caricati {len(problems)} problemi reali.\n")

    records: list = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(synergy_one, p["prompt"], p.get("expected", [])): p for p in problems}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # mai bloccare la suite
                r = {"cot_orig_acc": 0.0, "cot_red_acc": 0.0, "cpptai_red_acc": 0.0,
                     "sinergia_ok": 0, "orig_len": len(p["prompt"]), "red_len": 0,
                     "time_sec": 0.0, "error": str(e)[:200]}
            r["problem_id"] = p["id"]
            r["category"] = p.get("metadata", {}).get("category", "unknown")
            records.append(r)
            n_ok = sum(1 for x in records if x["sinergia_ok"])
            print(f"  {i}/{len(problems)} | sinergia={n_ok}/{i} | {p['id']} "
                  f"cot_orig={r['cot_orig_acc']} cot_red={r['cot_red_acc']} cpptai_red={r['cpptai_red_acc']} "
                  f"len {r['orig_len']}->{r['red_len']} {r['time_sec']}s", flush=True)

    def mean(k: str) -> float:
        return round(sum(x.get(k, 0.0) for x in records) / max(1, len(records)), 3)

    summary = {
        "n": len(records),
        "cot_originale": mean("cot_orig_acc"),
        "cot_ridotto": mean("cot_red_acc"),
        "cpptai_ridotto": mean("cpptai_red_acc"),
        "sinergia": mean("sinergia_ok"),
        "riduzione_media_ch": round(sum(x["orig_len"] - x["red_len"] for x in records) / max(1, len(records)), 1),
        "tempo_totale_s": round(time.perf_counter() - t0, 1),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("benchmarks/synergy")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"synergy_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["problem_id", "category", "cot_orig_acc", "cot_red_acc",
                                          "cpptai_red_acc", "sinergia_ok", "orig_len", "red_len", "time_sec"])
        w.writeheader()
        for r in sorted(records, key=lambda x: x["problem_id"]):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    json_path = outdir / f"summary_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== RISULTATO ({summary['tempo_totale_s']}s) ===")
    print(f"CoT originale: {summary['cot_originale']} | CoT ridotto: {summary['cot_ridotto']} | "
          f"CPPTAI ridotto: {summary['cpptai_ridotto']} | SINERGIA: {summary['sinergia']}")
    print(f"Riduzione media: {summary['riduzione_media_ch']} caratteri/problema")
    print(f"CSV: {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
