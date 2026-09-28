# DIARIO — CPPTAI V2 (nuovo diario di progetto)

**Data di apertura nuovo diario:** 28 Settembre 2026
**Autore:** analisi del codice attuale + continuazione di `testfatti/cosafatto.md`
**Stato progetto:** molto più avanti dei piani Explorer/Analyzer del 23/06/2026 — il vecchio diario si fermava al Giorno 5 (25/06 notte, N=200 confermato). Questo diario riparte da lì.

> Documenti precedenti, già letti e assorbiti:
> - `PLAN_EXPLORER_ANALYZER.md` (23/06, agent-goal-planner) — piano strategico M1-M5, CONDITIONAL GO.
> - `docs/explorer_analyzer_coordination_plan.md` (23/06, Agent-Based Coordinator) — piano tecnico Phase 0 / 0.5, gate, rollback, 5 settimane.
> - `testfatti/cosafatto.md` (23-25/06, Giorni 1-5) — implementazione Explorer+Analyzer, fix 5 fasi, baselines reali, Phase III dual-track, N=200 confermato.
> - `riasuntoccpatai_v2.md` — log grezzo sessione 23/06.

---

## 1. Dove siamo davvero oggi (28/09/2026)

### 1.1 Timeline reale da file system (escluso `.opencode`, `__pycache__`, `.cache`)

| Data | Cosa è successo (da LastWriteTime) |
|------|------------------------------------|
| 23/06/2026 | Piani Explorer+Analyzer (i 2 file che mi hai fatto leggere). Analisi 3 agenti, score 47/100. |
| 23-24/06 | Implementazione Explorer+Analyzer + scale-up (batch, adattivo, 16 lenti, caching). |
| 24/06 | Giorno 2 Fondamenta: API key rimossa, Dockerfile hardenato, 17 bare-except fixati, Phase I Shannon reale, Phase II Kahn, Phase III beam search (prima versione), Phase IV 7 domini, Phase V 3 formati, baselines reali. 47/47 test. |
| 25/06 mattina | Bug fix Opus (phase0 params, token_counts, emoji cp1252), ResponsibleAI estratto, primi GSM8K 10 prob (70% vs 80% EA). |
| 25/06 13:20-13:26 | GSM8K 100 base 86% + GSM8K 20 EA 85% → scoperta: Explorer non aggiunge nulla su pipeline fixata. ~120 run JSON in `testfatti/`. |
| 25/06 17:18-17:35 | `exploration.py`, `humaneval_executor.py`, `deepseek_client.py` ritoccati. `requirements-lock.txt` (648 righe) pinnato. `run_baselines_gsm8k.py` creato + `benchmarks/baselines_gsm8k/comparison.json`. |
| 25/06 18:01-18:14 | Giorno 4: Phase III resa REALE dual-track (`core.py`, `phases/phase3.py`, `run_full_suite.py` con `--gsm8k-method` / `--compare-pipeline`, test aggiornati). |
| 25/06 18:15-21:13 | N=50 (90% bypass vs 98% pipeline, p=0.22 n.s.) poi **N=200 CONFERMATO: bypass 86.5% vs pipeline 96.0%, +9.5pp, McNemar exact p=0.00055**. `mcnemar_n200.py` + `benchmarks/full_suite/full_benchmarks_20260625_211350.csv` + `compare_n200b.log`. `cosafatto.md` chiuso alle 21:17. |
| 02/08/2026 20:03-20:12 | Giorno 6 (ricostruito): `research.tex` (188 righe) + `preprint_cpptai_v2_2026-08-02.tex` (530 righe) scritti. `memoria.json` + `ragionamenti.csv` rigenerati (run singolo). Nessun cambio codice. |
| 28/09/2026 | Giorno 7 (oggi): analisi completa codice + apertura di questo nuovo diario. Nessun run dal 02/08. |

**Conclusione:** dal 25/06 notte il codice è fermo. L'unico avanzamento dopo il Giorno 5 è il paper del 02/08. Il progetto NON è andato avanti sul codice — è andato avanti sulla carta. Il diario mancava proprio per il periodo 26/06 → 02/08 → oggi.

### 1.2 Mappa codice attuale (righe reali di oggi)

```
src/cpptai/
  core.py                 1299 righe — orchestratore + 5 fasi (EntropicSegregator, VerticalTopology,
                                          DescentVector dual-track, ConvergenceProtocol, ComplexityScorer,
                                          SemanticGradient, ConsistencyEnforcer, CPPTAITraslocatore)
  exploration.py           593 righe — ExplorerEngine + TrajectoryAnalyzer (16 lenti, adaptive 3/8/15,
                                          ThreadPoolExecutor, noise 0.5x-1.5x, cache dict, batch)
  benchmarks.py            434 righe — run_benchmarks refactored in 8 funzioni, energy 50 + HF 5/set,
                                          shannon, paired t + Cohen d, 6 file output
  types.py                 207 righe — ProblemBlock, ExplorationTrajectory, AnalyzerPrepared,
                                          RunConfig (explorer/analyzer flags), RunArtifact, SolutionState
  pipeline_v2.py           185 righe — run() 7 fasi (0, 0.5, I-V), artifact JSON, env CPPTAI_DISABLE_PHASE*
  presentation.py          177 righe — executive/technical/public, confidence bar, attribution
  baselines.py             120 righe — CoT/ToT/GoT/ReAct REALI su DeepSeek API (non più stub)
  deepseek_client.py       115 righe — client OpenAI-compat, fallback modelli, temperature/max_tokens
  humaneval_executor.py    100 righe — subprocess + timeout 10s, AST check
  responsible_ai.py         73 righe — ResponsibleAIAuditor (estratto da core.py il 25/06)
  datasets.py              220 righe — GSM8K 1319 streaming HF + categorizzazione, MATH 3 fallback, stub
  env.py                    36 righe — loader .env senza dipendenze
  tasks.py                  21 righe
  io/cache.py               36 righe — cache disco per connettori
  phases/ phase0_explorer 38 / phase0_analyzer 36 / phase1 22 / phase2 21 / phase3 35 / phase4 28 / phase5 23
  connectors/, eval/       scheletri (1 riga __init__)
src/main.py                 73 righe — CLI --v2 --benchmark --offline --explorer --analyzer --explorer-trajectories
scripts/
  run_full_suite.py        376 righe — BenchmarkOrchestrator, --gsm8k-method {bypass,pipeline},
                                          --compare-pipeline, --explorer, ThreadPoolExecutor
  run_gsm8k_massive.py     ~180 righe — massivo 1319
  run_baselines_gsm8k.py   116 righe — CPPTAI vs 4 baselines su GSM8K (25/06)
  run_suite.py             legacy
configs/ default.yaml, offline.yaml, with_explorer.yaml (5 traj, noise 0.6, temp 0.8, denoise 3)
tests/ 9 file — test_core, test_explorer (24 test), test_explorer_analyzer_integration (4),
                 test_humaneval (4), test_benchmark_automation (4), test_presentation (2),
                 test_properties (3), test_v2_pipeline (2), check_benchmarks
```

### 1.3 Numeri definitivi (quelli che contano)

**Benchmark offline energy (root `benchmarks.csv`, `benchmarks_summary.csv`):**
CoT 0.0 / ToT 0.072 / GoT 0.0 / ReAct 0.072 / **CPPTAI 0.722** — diversity ~0.99, robust 0.82, 3 cluster. Paired t CPPTAI vs baselines 13.587, d ~2.0-2.2, p=0.0. Ablation CPPTAI vs no_IV vs no_I identiche (0.722) → ablation non discriminante su energy, problema noto dal Giorno 1 mai risolto.

**GSM8K N=200 appaiato (25/06 21:13, `full_benchmarks_20260625_211350.csv`, verificato con `mcnemar_n200.py`):**
bypass `solve_gsm8k` single-shot **86.5%** (173/200, 40s) vs pipeline `solve()` dual-track FLOORS=2/BRANCHING=2 **96.0%** (192/200, 410s). Discordanti 29: b01=24 recuperi, b10=5 rotture, **McNemar exact two-sided p=0.00055, p<0.001**. È il risultato più solido del progetto.

**Anomalia aperta — baselines GSM8K N=20 (`benchmarks/baselines_gsm8k/comparison.json`, 25/06 17:34):**
CPPTAI 20.0% (4/20) vs CoT 100% / ReAct 100% / ToT 95% / GoT 85%. Causa probabile: `run_baselines_gsm8k.py` chiama `solve_gsm8k` su orchestratore fresco ma confronta con `extract_number` diverso da `gsm8k_accuracy`, oppure i 20 problemi sono i primi dello streaming (i più facili per i baseline verbosi, i più fragili per il single-shot secco). NON è stato mai investigato. È il task C1 della roadmap 5 agenti rimasto a metà: le baselines non sono mai state rilanciate su N=100 con lo stesso scorer del compare-pipeline.

**Explorer su GSM8K:** 70→80% su pipeline debole (Giorno 1), 86→85% su pipeline fixata (Giorno 3). L'ablation trajectories 3/5/10 e denoising 1/3/5 (I3/I4) non è mai stata eseguita.

---

## 2. Cosa è stato fatto DOPO il vecchio diario (ricostruzione Giorno 6)

Il vecchio diario chiude al Giorno 5 notte. Ecco il Giorno 6, ricostruito dai file:

### Giorno 6 — 2 Agosto 2026: il paper

1. **`research.tex` (188 righe)** — documento ricerca: abstract 96.0% vs 86.5% p=0.00055, design sistema, artifact canonico, modalità online/cached/offline, dual-track FLOORS=2/BRANCHING=2, 2 figure pgfplots (energy 72.2% + GSM8K N=200), tabella McNemar b11=168/b10=5/b01=24/b00=3, CI GitHub Actions, lavori futuri (ricablare bypass, FLOORS=3, Phase IV reale, MATH/HumanEval, latenza 10x).
2. **`preprint_cpptai_v2_2026-08-02.tex` (530 righe)** — preprint completo: intro CoT/ToT/GoT/ReAct con citazioni, metodologia multi-agente (3+5 agenti, 27 rilievi, 47→86), Explorer/Analyzer, Phase III dual-track 4 lenti, Responsible AI, artifact, benchmark, statistica. È il documento che incorpora `riasuntoccpatai_v2.md` + `cosafatto.md`.
3. **Run singolo 02/08 20:03** — `memoria.json` (12KB) + `ragionamenti.csv` rigenerati. Probabilmente `python src/main.py` di verifica pre-paper. Nessun benchmark rilanciato.
4. **NON fatto il 02/08:** nessun fix codice, nessun N=1319, nessun MATH/HumanEval, nessuna ablation Explorer, nessuna investigazione anomalia baselines N=20.

---

## 3. Analisi onesta del codice di oggi (28/09)

### 3.1 Cosa è solido
- **Phase III dual-track** (`core.py` + `phases/phase3.py`): track float invariato (attribution/counterfactual/log preservati) + track testuale con 4 lenti verify/complete/concretize/simplify, ground-floor `Final answer:`, gradiente sul testo reale, `offline=` + `CPPTAI_OFFLINE=1` + `CPPTAI_DESCENT_FLOORS/BRANCHING`. È l'unica fase che risolve davvero problemi. Prova N=200.
- **Pipeline v2** (`pipeline_v2.py`): 7 fasi con flag env, artifact JSON per run (`artifacts/` 13 run + `testfatti/` ~120 run), backward-compat verificata dai test.
- **Explorer/Analyzer** (`exploration.py` 593 righe): 16 lenti, adaptive, parallelo, cache, batch. Test ermetici con `offline_mode`. Non aggiunge accuracy su pipeline forte, ma è innovativo e integro.
- **Baselines reali** (`baselines.py`): 4 classi su DeepSeek API. Mai più stub.
- **Determinismo**: modalità offline/cached, seed, cache normalizzata, suite ermetica 47/47 con `DEEPSEEK_API_KEY="" + CPPTAI_OFFLINE=1` (~31s).
- **Paper**: 2 .tex compilabili, figure/tabelle agganciate ai CSV veri.

### 3.2 Debiti aperti (confermati oggi, ereditati dalla roadmap 5 agenti del 25/06)
1. **Anomalia baselines GSM8K N=20** (CPPTAI 20% vs CoT 100%) — mai investigata. Priorità #1: rilanciare `run_baselines_gsm8k.py --n 100` con stesso scorer di `run_full_suite.py`.
2. **GSM8K FULL 1319 mai lanciato** — `run_gsm8k_massive.py` pronto, mai eseguito. 10 min stimati.
3. **MATH mai testato** — 3 fallback in `datasets.py` (hendrycks → competition_math → stub sintetico, il log compare_n200b conferma che i primi 2 falliscono). HumanEval infra pronto (`humaneval_executor.py` subprocess non sandboxato — task C3/S2 ancora aperto: serve `python -I` + env minimale) ma mai eseguito su 20-50 problemi.
4. **Ablation Explorer** (trajectories 3/5/10, denoising 1/3/5) mai eseguita. Makefile ha già `benchmark-explorer`.
5. **Latenza 10x** (0.2s bypass vs ~2s pipeline, 14.5s con EA) — cache persistente shelve/gzip (O1) mai fatta. Cache attuale è dict in-memory.
6. **Sicurezza**: API key `sk-82db3e3...` citata in chiaro nel vecchio diario (ruotare, C2/S1). `requirements-lock.txt` 648 righe inquinato da editable locali (`alpha-omega-llm-good`, `anima`, `alphagenome`, path `C:/Users/acese/...`) — da rigenerare da env pulito (I5/S3). Retry/backoff 429 (I6/S5), HEALTHCHECK finto (I7/S4), `.dockerignore` vs `scripts/` (I8), `long_term_memory` senza cap (I9), `sys.path.insert(0,...)` (O6) — tutti ancora aperti, verificati oggi nel codice.
7. **Copertura ~27-30%** — `eval/` e `connectors/` scheletri, `tasks.py` 21 righe, nessun test per MATH/HumanEval su larga scala.
8. **Ablation energy non discriminante** (CPPTAI = no_IV = no_I = 0.722) — il problema "fasi identiche = non pubblicabile" del Giorno 1 è rientrato dalla finestra su energy. Su GSM8K pipeline-vs-bypass invece discrimina.

### 3.3 Punteggio aggiornato (onesto)
```
25/06 notte:  86/100 — "86% GSM8K 100, pipeline robusta" (ma con N=200 poi 96% pipeline reale)
02/08 paper:  90/100 — "+ preprint + research.tex, risultato N=200 blindato"
oggi 28/09:  90/100 — codice fermo al 25/06, paper fermo al 02/08, debiti invariati
target:      ~96/100 — con anomalia baselines chiarita + FULL 1319 + MATH/HumanEval + ablation EA + lock pulito
```

---

## 4. Prossimi passi (roadmap ripresa, non riscritta)

**Wave 1 — Verità (~1 ora, priorità assoluta):**
- [ ] Rilanciare baselines GSM8K N=100 con scorer unico (`run_baselines_gsm8k.py` vs `run_full_suite.py --compare-pipeline`) e chiarire l'anomalia 20% vs 100%.
- [ ] Ruotare API key DeepSeek (citata in chiaro nel vecchio diario).
- [ ] Sandboxare `humaneval_executor.py` (`python -I` + env minimale + timeout).
- [ ] GSM8K FULL 1319 solo pipeline (`run_gsm8k_massive.py --max-problems 1319 --workers 4`).

**Wave 2 — Completezza (~1 ora):**
- [ ] MATH 100 (verificare quale dei 3 loader sopravvive).
- [ ] HumanEval 20-50 problemi sandboxato.
- [ ] Ablation Explorer trajectories 3/5/10 + denoising 1/3/5 su 50 problemi.
- [ ] Rigenerare `requirements-lock.txt` da ambiente pulito.

**Wave 3 — Paper upgrade:**
- [ ] Ricablare benchmark GSM8K ufficiale fuori dal bypass (task aperto dal Giorno 5 §12E).
- [ ] FLOORS=3/BRANCHING=3 su 100 problemi (il gap cresce?).
- [ ] Phase IV con fonti reali (oggi `enable_phase_iv=False` nei run GSM8K).
- [ ] Aggiornare `research.tex` + preprint con FULL 1319 + MATH + HumanEval.

---

## 5. Nota di metodo (per chi riprende)

- I piani del 23/06 (M1-M5, 5 settimane, coordinator, gate, degradation detector) sono **superati nei fatti**: M1-M3 implementati in `exploration.py` + `phases/phase0_*` + `pipeline_v2.py`, ma senza `coordinator.py`/`degradation.py` separati e senza gate formali — il fallback è inline nei phase wrapper. Non ha senso "tornare indietro" a quella struttura: il codice funziona e i test lo provano.
- Il vecchio diario resta fonte primaria per 23-25/06. Questo diario NON lo riscrive, lo continua. Prossima voce: data del prossimo run, non della prossima analisi.
- Regola del diario d'ora in poi: **ogni run >20 problemi aggiunge una riga qui con data, comando, N, accuracy, tempo, CSV**. Niente più salti di 2 mesi senza diario.

## 5. Giorno 7 — 28/09/2026: sinergia CoT+CPPTAI (5 step, N=20 reali)

Su richiesta utente, nuovo modo di verificare che CoT e CPPTAI funzionino INSIEME:
`tests/test_cot_cpptai_synergy.py` (5 test, pipeline pensiero→analisi→meta→riduzione→output CoT+CPPTAI
sullo stesso input ridotto) + `scripts/run_synergy_suite.py` (suite su N problemi).

### 5A. Unit test: 5/5 OK in ~41s (API reale, problema Natalia atteso 72)
CoT=1.0 e CPPTAI=1.0 sul ridotto (709ch → 132ch). Skip automatico senza `DEEPSEEK_API_KEY`.

### 5B. Suite N=20 GSM8K reali (`benchmarks/synergy/synergy_20260928_164222.csv`, 333.5s, workers=2)
| Metrica | Valore |
|---|---|
| CoT su originale | **1.000** (20/20) |
| CoT su ridotto | **0.950** (19/20) |
| CPPTAI su ridotto | **0.900** (18/20) |
| SINERGIA (almeno uno risolve) | **0.950** (19/20) |
| Riduzione media | 86.0 caratteri/problema |

### 5C. Lettura onesta
- Primi 20 GSM8K sono facili per CoT single-shot (100%): confermano che il confronto baselines vada fatto su N=100+,
  non sui primi 20.
- La riduzione toglie ~86ch ma costa ~5%: gsm8k_17 rotto per entrambi dopo la riduzione, gsm8k_8 rotto solo per CPPTAI.
  La sinergia (OR dei due) recupera quasi tutto: 19/20.
- Prossimo: stessa suite su N=100 per numeri solidi + ablation con/senza riduzione.

---
*Nuovo diario aperto il 28/09/2026 dopo analisi completa del codice. Prossimo aggiornamento al primo run utile.*
