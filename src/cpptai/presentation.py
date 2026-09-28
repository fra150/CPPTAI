"""Phase V: Presentation and arrangement of the final solution.

Provides structured output formatters for different audiences
(executive, technical, public). Each produces a complete document
with summary, findings, recommendations, evidence, and confidence.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Core presentation function
# ---------------------------------------------------------------------------

def arrange_solution_simple(
    text: str,
    context: str = "technical",
    confidence: Optional[float] = None,
    attribution: Optional[str] = None,
    counterfactual: Optional[str] = None,
) -> str:
    """Format the solution for a target audience.

    Args:
        text: Raw solution text (final synthesis + external context).
        context: One of {"executive", "technical", "public"}.
        confidence: Optional confidence score [0,1] to include.
        attribution: Optional attribution explanation text.
        counterfactual: Optional counterfactual analysis text.
    """
    parts = _split_sections(text)

    template = {
        "executive": _format_executive,
        "technical": _format_technical,
        "public": _format_public,
    }

    formatter = template.get(context, _format_technical)
    return formatter(parts, confidence, attribution, counterfactual)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _split_sections(text: str) -> Dict[str, str]:
    """Split raw text into logical sections: summary, findings, evidence."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text_flat = " ".join(lines) if not lines else "\n".join(lines)

    # Try to split on [label] markers from Phase IV synthesis
    sections: Dict[str, str] = {"summary": "", "findings": "", "evidence": ""}

    if "[Web]" in text_flat or "[DeepSeek]" in text_flat:
        # Structured synthesis — split by source labels
        source_blocks = re.split(r'\[(\w+)\]', text_flat)
        # source_blocks: [empty?] label1, content1, label2, content2, ...
        findings_parts = []
        evidence_parts = []
        for i in range(1, len(source_blocks) - 1, 2):
            label = source_blocks[i]
            content = source_blocks[i + 1].strip()
            if label in ("Web", "Science"):
                evidence_parts.append(f"- [{label}] {content}")
            else:
                findings_parts.append(f"- [{label}] {content}")
        sections["findings"] = "\n".join(findings_parts) if findings_parts else ""
        sections["evidence"] = "\n".join(evidence_parts) if evidence_parts else ""
        sections["summary"] = text_flat[:200] if len(text_flat) > 200 else text_flat
    else:
        # Flat text — split by length
        words = text_flat.split()
        if len(words) > 100:
            sections["summary"] = " ".join(words[:30])
            sections["findings"] = " ".join(words[30:70])
            sections["evidence"] = " ".join(words[70:])
        else:
            sections["summary"] = text_flat

    return sections


def extract_key_points(text: str) -> str:
    """Extract key points: first 3 substantive sentences."""
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip() and not p.isdigit()]
    points = []
    for p in parts:
        if len(p.split()) > 3:  # skip fragments
            points.append(p)
        if len(points) >= 3:
            break
    return "\n".join(f"- {p}" for p in (points or ["No key points extracted"]))


def extract_actions(text: str) -> str:
    """Extract action items from imperative-like phrases."""
    action_verbs = {
        "implement", "reduce", "evaluate", "deploy", "monitor", "develop",
        "create", "establish", "optimize", "integrate", "design", "build",
        "test", "validate", "scale", "improve", "expand", "launch",
    }
    candidates = []
    for token in text.split():
        if token.lower() in action_verbs:
            candidates.append(token)
    if not candidates:
        return "- Define next steps\n- Assign owners\n- Set timeline\n- Monitor outcomes"
    return "\n".join(f"- {c.title()} key measures" for c in candidates[:4])


def extract_conclusion(text: str) -> str:
    """Extract conclusion preferring last substantive paragraph."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    parts = [p for p in parts if not p.isdigit() and len(p.split()) > 3]
    return parts[-1] if parts else text


# ---------------------------------------------------------------------------
# Audience-specific formatters
# ---------------------------------------------------------------------------

def _format_executive(
    parts: Dict[str, str],
    confidence: Optional[float] = None,
    attribution: Optional[str] = None,
    counterfactual: Optional[str] = None,
) -> str:
    """Executive summary format — brevity and action."""
    lines = [
        "## Executive Summary",
        "",
        parts.get("summary", "No summary available."),
        "",
        "### Key Points",
        extract_key_points(parts.get("findings", parts.get("summary", ""))),
        "",
        "### Recommended Actions",
        extract_actions(parts.get("findings", "")),
    ]
    if confidence is not None:
        bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        lines += ["", f"### Confidence: {confidence:.0%}", f"`{bar}` {confidence:.0%}"]
    if attribution:
        lines += ["", "### Attribution", attribution[:300]]
    return "\n".join(lines)


def _format_technical(
    parts: Dict[str, str],
    confidence: Optional[float] = None,
    attribution: Optional[str] = None,
    counterfactual: Optional[str] = None,
) -> str:
    """Technical report format — structured and detailed."""
    lines = [
        "## Solution Report",
        "",
        "### Summary",
        parts.get("summary", "No summary available."),
        "",
        "### Analysis & Findings",
        parts.get("findings", "No findings extracted."),
        "",
        "### Supporting Evidence",
        parts.get("evidence", "No evidence available."),
        "",
        "### Conclusion",
        extract_conclusion(parts.get("summary", "")),
    ]
    if confidence is not None:
        lines += ["", f"### Confidence Score\n{confidence:.1%}"]
    if attribution:
        lines += ["", "### Attribution\n" + attribution]
    if counterfactual:
        lines += ["", "### Counterfactual Analysis\n" + counterfactual]
    lines += ["", "### Key Points", extract_key_points(parts.get("summary", ""))]
    return "\n".join(lines)


def _format_public(
    parts: Dict[str, str],
    confidence: Optional[float] = None,
    attribution: Optional[str] = None,
    counterfactual: Optional[str] = None,
) -> str:
    """Public-facing format — accessible and clear."""
    lines = [
        "## Solution Overview",
        "",
        parts.get("summary", "We found a solution to the problem."),
        "",
        "### What We Found",
        extract_key_points(parts.get("findings", parts.get("summary", ""))),
        "",
        "### What To Do Next",
        extract_actions(parts.get("findings", "")),
    ]
    return "\n".join(lines)
