"""Sinergia CoT + CPPTAIv2 — pipeline a 5 step con chiamate API reali.

Nuovo modo di verificare che CoT e CPPTAI funzionino INSIEME:
  Step 1 — Pensiero: l'AI pensa al problema e produce pensieri grezzi sui dati.
  Step 2 — Analisi dati: dai pensieri si estraggono dati strutturati.
  Step 3 — Analisi del pensiero: meta-valutazione della qualita del pensiero.
  Step 4 — Riduzione: si eliminano i dati che non servono, resta l'essenziale.
  Step 5 — Output: sul problema ridotto girano SIA CoT SIA CPPTAI e si confrontano.

Richiede DEEPSEEK_API_KEY (altrimenti skip). Uso: python -m unittest tests.test_cot_cpptai_synergy -v
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.deepseek_client import deepseek_chat, extract_text_answer
from cpptai.env import load_env
from cpptai.baselines import CoTBaseline
from cpptai.core import CPPTAITraslocatore
from cpptai.benchmarks import gsm8k_accuracy

# Problema fisso, semplice, risposta nota: 72. Costo API minimo, risultato stabile.
PROBLEM = (
    "Natalia sold 48 clips in May. In June she sold half as many clips as in May. "
    "How many clips did Natalia sell altogether in May and June?"
)
EXPECTED = ["72"]


def has_api_key() -> bool:
    load_env()
    return bool((os.getenv("DEEPSEEK_API_KEY") or "").strip())


def ask_ai(system: str, user: str, max_tokens: int = 512) -> str:
    """Singola chiamata API, fallisce forte se l'API non risponde (test veri)."""
    resp = deepseek_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
    )
    text = extract_text_answer(resp) if resp else ""
    return (text or "").strip()


# ---------------------------------------------------------------- 5 step
def step1_pensiero(problem: str) -> str:
    """Step 1 — Pensiero: l'AI ragiona a ruota libera sui dati del problema."""
    return ask_ai(
        "You are a careful thinker. Write down every raw thought and observation about the problem data.",
        f"Think aloud about this problem, listing all data, quantities and relations you notice:\n\n{problem}",
    )


def step2_analisi_dati(pensieri: str) -> str:
    """Step 2 — Analisi dei dati: dai pensieri grezzi ai dati strutturati."""
    return ask_ai(
        "You are a data analyst. Extract structured facts only, no chatter.",
        f"From these raw thoughts, extract the structured data (entities, numbers, relations, question asked). "
        f"One fact per line starting with '-'.\n\nThoughts:\n{pensieri}",
    )


def step3_analisi_pensiero(pensieri: str, analisi: str) -> str:
    """Step 3 — Analisi del pensiero: meta-giudizio sulla qualita del ragionamento."""
    return ask_ai(
        "You are a meta-cognition auditor. Judge the thinking quality briefly and honestly.",
        f"Raw thoughts:\n{pensieri}\n\nData analysis:\n{analisi}\n\n"
        f"Answer in 3 short lines: 1) is the thinking coherent? 2) is anything missing or wrong? 3) verdict: SOLID or WEAK.",
    )


def step4_riduzione(problem: str, analisi: str, meta: str) -> str:
    """Step 4 — Riduzione: elimina i dati che non servono, resta l'essenziale."""
    reduced = ask_ai(
        "You are a minimalist editor. Keep only what is needed to solve the problem. Reply with the reduced problem only.",
        f"Original problem:\n{problem}\n\nStructured data:\n{analisi}\n\nMeta verdict:\n{meta}\n\n"
        f"Rewrite the problem keeping ONLY the data needed to solve it. Drop everything else.",
    )
    return reduced


def step5_output(ridotto: str) -> dict:
    """Step 5 — Output: CoT e CPPTAI lavorano INSIEME sullo stesso input ridotto."""
    cot_answer = CoTBaseline().solve(ridotto)
    cpptai_result = CPPTAITraslocatore(enable_phase_iv=False).solve(ridotto)
    cpptai_answer = cpptai_result.get("final_answer", "")
    return {
        "cot": cot_answer,
        "cpptai": cpptai_answer,
        "cot_acc": gsm8k_accuracy(cot_answer, EXPECTED),
        "cpptai_acc": gsm8k_accuracy(cpptai_answer, EXPECTED),
        "sinergia_ok": bool(
            gsm8k_accuracy(cot_answer, EXPECTED) == 1.0
            or gsm8k_accuracy(cpptai_answer, EXPECTED) == 1.0
        ),
    }


def extract_numbers(text: str):
    return re.findall(r"-?\d+\.?\d*", text.replace(",", ""))


class TestCotCpptaiSynergy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not has_api_key():
            raise unittest.SkipTest("DEEPSEEK_API_KEY mancante: servono chiamate API reali.")

    def test_1_pensiero_produce_testo_sui_dati(self):
        pensieri = step1_pensiero(PROBLEM)
        self.assertTrue(len(pensieri) > 30, f"Pensiero troppo corto: {pensieri!r}")
        self.assertIn("48", pensieri, "Il pensiero deve contenere i dati chiave (48).")
        print(f"\n[STEP1 pensiero {len(pensieri)}ch]: {pensieri[:300]}")

    def test_2_analisi_estrapola_dati(self):
        pensieri = step1_pensiero(PROBLEM)
        analisi = step2_analisi_dati(pensieri)
        righe = [l for l in analisi.splitlines() if l.strip().startswith("-")]
        self.assertGreaterEqual(len(righe), 2, f"Analisi deve estrarre >=2 fatti:\n{analisi}")
        self.assertIn("48", analisi)
        print(f"\n[STEP2 analisi]:\n{analisi[:500]}")

    def test_3_meta_giudica_il_pensiero(self):
        pensieri = step1_pensiero(PROBLEM)
        analisi = step2_analisi_dati(pensieri)
        meta = step3_analisi_pensiero(pensieri, analisi)
        self.assertTrue(len(meta) > 10, "Meta-analisi vuota.")
        self.assertTrue(
            ("SOLID" in meta.upper()) or ("WEAK" in meta.upper()),
            f"La meta-analisi deve dare un verdetto:\n{meta}",
        )
        print(f"\n[STEP3 meta]:\n{meta[:400]}")

    def test_4_riduzione_tiene_essenziale(self):
        pensieri = step1_pensiero(PROBLEM)
        analisi = step2_analisi_dati(pensieri)
        meta = step3_analisi_pensiero(pensieri, analisi)
        ridotto = step4_riduzione(PROBLEM, analisi, meta)
        self.assertTrue(len(ridotto) > 10, "Problema ridotto vuoto.")
        self.assertLess(
            len(ridotto), len(pensieri) + len(analisi),
            "La riduzione deve eliminare dati, non aggiungerne.",
        )
        for numero in ("48",):
            self.assertIn(numero, ridotto, f"La riduzione non deve perdere i dati chiave ({numero}).")
        print(f"\n[STEP4 ridotto {len(ridotto)}ch da {len(pensieri) + len(analisi)}ch]: {ridotto[:400]}")

    def test_5_cot_e_cpptai_insieme(self):
        pensieri = step1_pensiero(PROBLEM)
        analisi = step2_analisi_dati(pensieri)
        meta = step3_analisi_pensiero(pensieri, analisi)
        ridotto = step4_riduzione(PROBLEM, analisi, meta)
        out = step5_output(ridotto)
        print(f"\n[STEP5 CoT]: {out['cot'][:300]}")
        print(f"[STEP5 CPPTAI]: {out['cpptai'][:300]}")
        print(f"[STEP5 acc] CoT={out['cot_acc']} CPPTAI={out['cpptai_acc']}")
        self.assertTrue(out["cot"], "CoT non ha prodotto output.")
        self.assertTrue(out["cpptai"], "CPPTAI non ha prodotto output.")
        self.assertTrue(
            out["sinergia_ok"],
            f"Nessuno dei due ha risolto il problema ridotto (atteso 72). CoT={out['cot_acc']} CPPTAI={out['cpptai_acc']}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
