# CPPTAI

Python framework (standard library only) for a 5‑phase cognitive pipeline, DeepSeek API integration (OpenAI‑compatible), automated benchmarks with CSV/JSON reports, and LaTeX research generation with auto‑loaded tables and plots.

## Overview
- Five‑phase architecture: Entropic segregation → Vertical topology → Cognitive descent → External convergence → Presentation (Phase V).
- Minimal DeepSeek client with `.env` key management and model fallback.
- Benchmarks: accuracy vs baselines (CoT, ToT, GoT, ReAct), diversity (Shannon on clusters), error rates (GSM8K/MATH/AIME), time‑per‑problem; outputs in `benchmarks.csv` and `benchmarks.json`.
- LaTeX research (`research.tex`) using `pgfplotstable`/`pgfplots` to load results directly.

## Requirements
- Python ≥ 3.10.
- No external dependencies; everything uses the standard library.
- Optional: DeepSeek API key for live responses.

## Setup
1. Create a `.env` file at the project root:
   
   ```
   DEEPSEEK_API_KEY=your_key
   ```
   
   Do not share or commit real keys.

2. Main file structure:
   - `src/cpptai/core.py` – 5‑phase orchestrator.
   - `src/cpptai/deepseek_client.py` – DeepSeek API client with `.env` loader.
   - `src/cpptai/benchmarks.py` – execution and metrics, CSV/JSON outputs.
   - `src/cpptai/presentation.py` – Phase V formatting (executive/technical/public).
   - `src/cpptai/types.py` – base types (difficulty, problem blocks).
   - `src/cpptai/env.py` – `.env` loader without dependencies.
   - `src/main.py` – startup CLI.
   - `tests/` – unit tests.
   - `research.tex` – LaTeX document (auto‑loaded charts/tables).

## Quick Start
- Run the main pipeline:
  - Windows PowerShell: `python .\src\main.py`
  - Alternative: `python -m src.main`

- Primary outputs:
  - `ragionamenti.csv` – pipeline artifacts.
  - `memoria.json` – execution state/memory.
  - `benchmarks.csv` and `benchmarks.json` – quantitative results.

## Benchmarks
- Implemented metrics:
  - Accuracy vs baselines: CoT, ToT, GoT, ReAct.
  - Diversity: Shannon entropy over solution clusters (hash embeddings + k‑means).
  - Error rates: GSM8K/MATH/AIME sets (proxy/placeholder if data unavailable).
  - Average time per problem and token counts.
- Execution: run `src/main.py`; benchmarks run automatically and are saved to `benchmarks.csv`/`benchmarks.json`.

## LaTeX Research
- `research.tex` loads `benchmarks.csv` with `pgfplotstable` and generates tables/plots.
- Typical compilation:
  - `pdflatex research.tex`
  - Repeat compilation if needed to update references.
- Requires a LaTeX distribution with `pgfplots`/`pgfplotstable` (e.g., TeX Live/MiKTeX).

## Tests
- Run unit tests:
  - `python -m unittest discover -s tests -p "test_*.py" -q`
- Tests use only the standard library’s `unittest`.

## Security Notes
- Do not commit API keys/secrets.
- The client avoids logging sensitive content and uses fallbacks when no key is present.

## Troubleshooting
- Import errors in tests: ensure `src` is on `PYTHONPATH` or use the commands above; tests already include a local path fix.
- No response from DeepSeek: verify `DEEPSEEK_API_KEY` in `.env` and connectivity.

## Availability
- Repository: https://github.com/fra150/CPPTAI

## Generate Benchmarks and Figures
- Disable external calls if needed: PowerShell `$env:BENCH_DISABLE_EXTERNAL=1; python .\src\main.py`
- Outputs:
  - `benchmarks.csv`, `benchmarks_summary.csv`, `benchmarks.json`
  - `cumulative_accuracy.csv`, `error_by_phase.csv`, `stats_summary.csv`
- Compile LaTeX: `pdflatex research.tex` (twice for references).

## License
- MIT open‑source
