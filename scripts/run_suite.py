from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, os.path.abspath("src"))

from cpptai.pipeline_v2 import run as run_v2
from cpptai.types import RunConfig


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data


def _build_run_config(data: Dict[str, Any]) -> RunConfig:
    cfg = RunConfig(
        seed=int(data.get("seed", 0)),
        external_enabled=bool(data.get("external_enabled", True)),
        cache_mode=str(data.get("cache_mode", "online")),
        output_dir=str(data.get("output_dir", ".")),
        benchmark_name=str(data.get("benchmark_name", "mixed")),
        model_name=str(data.get("model_name", "DeepSeek-V3.2-Exp")),
        max_iterations=int(data.get("max_iterations", 100)),
    )
    os.environ["CPPTAI_CACHE_MODE"] = cfg.cache_mode
    os.environ["CPPTAI_OUTPUT_DIR"] = cfg.output_dir
    os.environ["CPPTAI_BENCHMARK"] = cfg.benchmark_name
    os.environ["CPPTAI_MODEL"] = cfg.model_name
    os.environ["CPPTAI_MAX_ITERS"] = str(cfg.max_iterations)
    os.environ["CPPTAI_SEED"] = str(cfg.seed)
    cache_dir = str(data.get("cache_dir", ".cache"))
    os.environ["CPPTAI_CACHE_DIR"] = cache_dir
    if not cfg.external_enabled:
        os.environ["BENCH_DISABLE_EXTERNAL"] = "1"
        os.environ["CPPTAI_DISABLE_EXTERNAL"] = "1"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--gsm8k", type=int, default=0)
    ns = parser.parse_args()

    data = _load_config(ns.config)
    cfg = _build_run_config(data)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "suite_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    problem = data.get(
        "problem",
        (
            "How can we address the global energy crisis considering: "
            "1) limits of renewables, 2) nuclear costs, 3) fossil dependency, "
            "4) geopolitical factors, 5) a just transition for workers?"
        ),
    )

    result, artifact = run_v2(problem, cfg)
    print("Run ID:", artifact.run_id)
    print("Final Answer:", result.get("final_answer", ""))

    if ns.gsm8k and ns.gsm8k > 0:
        from cpptai.benchmarks import run_gsm8k_cpptai

        recs, summ = run_gsm8k_cpptai(n=int(ns.gsm8k))
        (Path(cfg.output_dir) / "gsm8k_eval.json").write_text(
            json.dumps({"summary": summ, "records": recs}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("GSM8K:", summ)

    if not ns.no_benchmark:
        from cpptai.benchmarks import run_benchmarks

        records, summary = run_benchmarks()
        (Path(cfg.output_dir) / "benchmarks_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Benchmarks:", summary)


if __name__ == "__main__":
    main()
