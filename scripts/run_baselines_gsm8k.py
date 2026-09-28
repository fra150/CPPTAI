"""Confronto rapido: CPPTAI vs CoT/ToT/GoT/ReAct su GSM8K.
Esempio: python scripts/run_baselines_gsm8k.py --n 20 --workers 4
"""
from __future__ import annotations
import argparse
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from cpptai.datasets import DatasetLoader
from cpptai.baselines import CoTBaseline, ToTBaseline, GoTBaseline, ReActBaseline
from cpptai.core import CPPTAITraslocatore
def solve_cpptai(orchestrator, problem_text: str) -> str:
    result = orchestrator.solve_gsm8k(problem_text)
    return result.get("final_answer", "")


def extract_number(answer: str) -> float | None:
    """Estrae il numero dopo '####' (formato GSM8K), altrimenti l'ultimo numero."""
    import re
    # Cerca il marker ####
    if "####" in answer:
        after = answer.split("####")[-1].strip()
        nums = re.findall(r"-?\d+\.?\d*", after.replace(",", ""))
        if nums:
            return float(nums[-1])
    # Fallback: ultimo numero in tutto il testo
    nums = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    if nums:
        return float(nums[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="CPPTAI vs Baselines su GSM8K")
    parser.add_argument("--n", type=int, default=20, help="Numero problemi GSM8K")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CPPTAI vs BASELINES su GSM8K ({args.n} problemi)")
    print(f"{'='*60}\n")

    loader = DatasetLoader()
    problems = loader.load_gsm8k(n=args.n)
    print(f"Caricati {len(problems)} problemi GSM8K\n")

    methods = {
        "CPPTAI": lambda p: solve_cpptai(CPPTAITraslocatore(), p),
        "CoT": lambda p: CoTBaseline().solve(p),
        "ToT": lambda p: ToTBaseline().solve(p),
        "GoT": lambda p: GoTBaseline().solve(p),
        "ReAct": lambda p: ReActBaseline().solve(p),
    }

    results = {name: {"correct": 0, "total": 0, "time": 0.0} for name in methods}

    for i, prob in enumerate(problems):
        prompt = prob["prompt"] if isinstance(prob, dict) else str(prob)
        expected = prob.get("answer", prob.get("expected", ""))

        print(f"[{i+1}/{len(problems)}] {prompt[:60]}...")

        for name, fn in methods.items():
            t0 = time.perf_counter()
            try:
                answer = fn(prompt)
            except Exception as e:
                answer = f"ERR: {e}"
            dt = time.perf_counter() - t0

            # Verifica se la risposta contiene il numero corretto
            exp_num = extract_number(str(expected))
            ans_num = extract_number(str(answer))
            is_correct = False
            if exp_num is not None and ans_num is not None:
                is_correct = abs(exp_num - ans_num) < 0.01

            results[name]["total"] += 1
            if is_correct:
                results[name]["correct"] += 1
            results[name]["time"] += dt

            status = "OK" if is_correct else "NO"
            print(f"    {name:8s}: {status} ({dt:.1f}s) -> {str(answer)[:50]}")

    print(f"\n{'='*60}")
    print(f"  RISULTATI FINALI")
    print(f"{'='*60}\n")
    print(f"{'Metodo':12s} {'Accuratezza':>12s} {'Tempo':>10s}")
    print(f"{'-'*40}")
    for name, data in sorted(results.items(), key=lambda x: -x[1]["correct"]/max(1,x[1]["total"])):
        acc = data["correct"] / max(1, data["total"]) * 100
        t = data["time"]
        print(f"{name:12s} {acc:6.1f}% ({data['correct']}/{data['total']}) {t:7.1f}s")

    # Salva report
    out_dir = Path("benchmarks/baselines_gsm8k")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_problems": args.n,
        "results": {k: {"correct": v["correct"], "total": v["total"],
                        "accuracy": round(v["correct"]/max(1,v["total"])*100, 1),
                        "time_sec": round(v["time"], 1)}
                    for k, v in results.items()}
    }
    report_path = out_dir / "comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport salvato: {report_path}")


if __name__ == "__main__":
    main()
