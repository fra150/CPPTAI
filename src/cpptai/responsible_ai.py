"""Responsible AI auditor — bias detection and fairness checks.

Extracted from core.py into its own module for clarity and testability.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


class ResponsibleAIAuditor:
    """Audits generated outputs for protected-attribute bias and negative framing.

    Maintains a vocabulary of protected terms (demographic attributes) and
    negative-context keywords. Scans text for co-occurrence patterns and
    returns a structured risk assessment.
    """

    def __init__(self):
        self.protected_terms = [
            "woman", "women", "man", "men", "male", "female",
            "girl", "boy", "black", "white", "asian", "latino",
            "hispanic", "arab", "jewish", "muslim", "christian",
            "gay", "lesbian", "bisexual", "trans", "transgender",
            "disabled", "autistic", "elderly", "old", "young",
            "immigrant",
        ]
        self.negative_terms = {
            "inferior", "superior", "lazy", "stupid", "criminal",
            "dangerous", "dirty", "illegal", "terrorist", "untrustworthy",
        }

    def audit_bias_detection(self, text: str) -> Dict[str, Any]:
        """Analyse *text* for protected-attribute mentions near negative terms.

        Returns a verdict dict with keys:
            verdict — "pass" | "review"
            risk_score — float in [0, 1]
            protected_attribute_mentions — sorted list of matched terms
            negative_context_hits — list of {term, negative_terms, context}
            flags — human-readable notes
        """
        lowered = text.lower()
        mentions: List[str] = []
        negative_hits: List[Dict[str, Any]] = []

        words = re.findall(r"[a-zA-Z']+", lowered)
        for idx, w in enumerate(words):
            if w not in self.protected_terms:
                continue
            mentions.append(w)
            start = max(0, idx - 6)
            end = min(len(words), idx + 7)
            window = words[start:end]
            hit_terms = sorted(set(t for t in window if t in self.negative_terms))
            if hit_terms:
                negative_hits.append({
                    "term": w,
                    "negative_terms": hit_terms,
                    "context": " ".join(window),
                })

        unique_mentions = sorted(set(mentions))
        unique_negative = len(negative_hits)
        risk = 0.0
        if unique_mentions:
            risk = 0.3
        if unique_negative:
            risk = min(1.0, risk + 0.2 * unique_negative)

        verdict = "pass"
        flags: List[str] = []
        if unique_negative:
            verdict = "review"
            flags.append("negative_context_near_protected_attribute")
        if not unique_mentions:
            flags.append("no_protected_attribute_mentions_detected")

        return {
            "verdict": verdict,
            "risk_score": round(risk, 3),
            "protected_attribute_mentions": unique_mentions,
            "negative_context_hits": negative_hits,
            "flags": flags,
        }
