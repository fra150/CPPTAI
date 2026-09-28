# CPPTAI V2

Python framework (standard library only) for a **7-phase cognitive pipeline**: Explorer (Phase 0) → Analyzer (Phase 0.5) → Entropic segregation (I) → Vertical topology (II) → Cognitive descent (III) → External convergence (IV) → Presentation (V), with DeepSeek API integration (OpenAI-compatible), automated benchmarks with CSV/JSON reports, Responsible AI audit, and LaTeX research generation.

**Headline result (25/06/2026, GSM8K N=200 real problems, paired): pipeline 96.0% vs DeepSeek single-shot 86.5% (+9.5pp, McNemar exact p = 0.00055, p < 0.001).**

## Overview
- Seven-phase architecture: diffusion-inspired **Explorer** (multi-trajectory generation, 16 semantic lenses, adaptive count, caching) → **Analyzer** (trajectory selection + problem enrichment) → Entropic segregation (real Shannon sliding-window) → Vertical topology (Kahn topological sort) → Cognitive descent (**dual-track beam search** with 4 semantic lenses) → External convergence (7 topic domains) → Presentation (executive/technical/public).
- Minimal DeepSeek client with `.env` key management, model fallback, retry, optional file cache.
- Real reasoning baselines that call the LLM: CoT, ToT, GoT, ReAct (`src/cpptai/baselines.py`).
- **NEW — 5-step CoT+CPPTAI synergy** (`tests/test_cot_cpptai_synergy.py`, `scripts/run_synergy_suite.py`): thinking → data analysis → thought meta-analysis → reduction → joint CoT+CPPTAI output. GSM8K N=20: synergy 0.95.
- Benchmarks: accuracy vs baselines, diversity (Shannon on clusters), error rates (GSM8K/MATH/HumanEval), time-per-problem; outputs in `benchmarks/` (+ `benchmarks.csv`/`benchmarks.json` legacy root copies).
- LaTeX research (`research.tex`, `preprint_cpptai_v2_2026-08-02.tex`) with pgfplots figures built from the real CSVs.
- Project diary: `DIARIO.md` (continues `testfatti/cosafatto.md`, Giorni 1-7).

## Requirements
- Python ≥ 3.10.
- No external dependencies for the core framework; everything uses the standard library.
- Optional: DeepSeek API key for live responses; `datasets` (HuggingFace) for real GSM8K/MATH loading (falls back to synthetic stubs).

## Setup
1. Create a `.env` file at the project root:

   ```
   DEEPSEEK_API_KEY=your_key
   ```

   Do not share or commit real keys (`.env` is git-ignored; rotate any key ever pasted in chat).

2. Main file structure:
   - `src/cpptai/core.py` – orchestrator (`CPPTAITraslocatore`) + phases I–V.
   - `src/cpptai/exploration.py` – `ExplorerEngine` + `TrajectoryAnalyzer` (phases 0/0.5).
   - `src/cpptai/phases/` – thin `run(state, config)` wrappers per phase.
   - `src/cpptai/pipeline_v2.py` – 7-phase runner with structured `RunArtifact` JSON.
   - `src/cpptai/deepseek_client.py` – DeepSeek API client with `.env` loader.
   - `src/cpptai/benchmarks.py` – metrics (rubric/gsm8k accuracy, Shannon, paired t + Cohen's d).
   - `src/cpptai/baselines.py` – real CoT/ToT/GoT/ReAct baselines.
   - `src/cpptai/datasets.py` – GSM8K (1319 streaming), MATH (3 fallbacks), HumanEval stubs.
   - `src/cpptai/humaneval_executor.py` – real subprocess-based HumanEval verifier.
   - `src/cpptai/presentation.py` – Phase V formatting (executive/technical/public).
   - `src/cpptai/responsible_ai.py` – bias audit on protected attributes.
   - `src/cpptai/types.py` – canonical dataclasses (incl. `ExplorationTrajectory`, `AnalyzerPrepared`).
   - `src/main.py` – startup CLI.
   - `scripts/run_full_suite.py` – unified benchmark orchestrator (GSM8K/MATH/HumanEval, `--compare-pipeline`).
   - `scripts/run_synergy_suite.py` – 5-step CoT+CPPTAI synergy suite.
   - `scripts/run_gsm8k_massive.py` – massive GSM8K runs (up to 1319).
   - `scripts/run_baselines_gsm8k.py` – CPPTAI vs baselines on GSM8K.
   - `configs/` – `default.yaml`, `offline.yaml`, `with_explorer.yaml`.
   - `tests/` – unit + integration tests (incl. synergy).
   - `DIARIO.md` – living project diary.

## Quick Start
- Run the main pipeline:
  - Windows PowerShell: `python .\src\main.py --v2 --explorer --analyzer`
  - Offline (no external calls): `python .\src\main.py --v2 --offline`
  - Flags: `--benchmark/--no-benchmark`, `--output-dir`, `--explorer-trajectories N`.

- Synergy 5-step suite (needs API key):
  - `python scripts/run_synergy_suite.py --n 20 --workers 2`

- Full benchmark suite:
  - `python scripts/run_full_suite.py --compare-pipeline --gsm8k 200`
  - `make benchmark` (50 GSM8K + MATH + HumanEval) · `make benchmark-full` (1319+100+164)

- Primary outputs:
  - `artifacts/run_<uuid>.json` – structured per-run artifacts.
  - `benchmarks/` – CSV/JSON reports per suite run.
  - `ragionamenti.csv`, `memoria.json` – legacy pipeline artifacts.

## Benchmarks
- Key results (all with real DeepSeek API):
  - **GSM8K N=200 paired**: bypass single-shot 86.5% vs pipeline (dual-track beam search, FLOORS=2/BRANCHING=2) **96.0%**, McNemar p=0.00055.
  - **Offline energy**: CPPTAI 72.2% vs CoT/ToT/GoT/ReAct 0–7.2% (paired t≈13.6, d≈2.0+).
  - **Synergy N=20**: CoT-original 1.00, CoT-reduced 0.95, CPPTAI-reduced 0.90, synergy 0.95 (avg −86 chars/problem).
- Metrics: rubric/gsm8k accuracy, Shannon diversity, robust diversity + clusters, pass@k, category breakdown, paired t-test + Cohen's d, McNemar for paired compares.
- Known trade-off: pipeline ≈10× latency of single-shot (~2s vs ~0.2s/problem; ~14.5s with Explorer).

## LaTeX Research
- `research.tex` and `preprint_cpptai_v2_2026-08-02.tex` load the real benchmark CSVs with `pgfplotstable`/`pgfplots`.
- Typical compilation (twice for references):
  - `pdflatex research.tex`
- Requires a LaTeX distribution with `pgfplots`/`pgfplotstable` (e.g., TeX Live/MiKTeX).

## Tests
- Run unit tests (hermetic, no network):
  - `$env:DEEPSEEK_API_KEY=""; $env:CPPTAI_OFFLINE="1"; python -m unittest discover -s tests -p "test_*.py" -q`
- API-backed tests (synergy) skip automatically without a key:
  - `python -m unittest discover -s tests -p "test_cot_cpptai_synergy.py" -v`
- 47 hermetic tests + 5 synergy tests, all green as of 28/09/2026.

## Security Notes
- Do not commit API keys/secrets (`.env` is ignored; never paste keys in chat without rotating them after).
- HumanEval executor runs LLM-generated code via subprocess — sandbox before untrusted use.
- The client avoids logging sensitive content and degrades gracefully with no key.

## Availability
- Repository: https://github.com/fra150/CPPTAI

## License
- MIT open-source
