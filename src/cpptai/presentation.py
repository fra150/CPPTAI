"""Phase V: Presentation and arrangement of the final solution.

Provides a simple formatter that tailors the output to different audiences
(executive, technical, public). This mirrors the MVP snippet in the source
document while keeping code and comments in English.
"""

from __future__ import annotations

from typing import Dict


def arrange_solution_simple(text: str, context: str = "technical") -> str:
    """Format the solution for a target audience.

    Args:
        text: Raw solution text (final synthesis).
        context: One of {"executive", "technical", "public"}.
    """
    template = {
        "executive": " **KEY POINTS**\n{key_points}\n\n **ACTIONS**\n{actions}",
        "technical": "## Analysis\n{analysis}\n\n## Solution\n{solution}\n\n## Details\n{details}",
        "public": "Hello!\nWe found a solution:\n\n{solution}\n\nWhat do you think?",
    }

    key_points = extract_key_points(text)
    actions = extract_actions(text)
    solution = extract_conclusion(text)
    analysis = text[:200]
    details = text

    return template.get(context, template["technical"]).format(
        key_points=key_points,
        actions=actions,
        analysis=analysis,
        solution=solution,
        details=details,
    )


def extract_key_points(text: str) -> str:
    """Naive key point extraction: first 3 sentences or bullet-like items."""
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return "\n".join(f"- {p}" for p in parts[:3])


def extract_actions(text: str) -> str:
    """Naive action extraction: look for imperative-like phrases."""
    tokens = text.split()
    candidates = [t for t in tokens if t.lower() in {"implement", "reduce", "evaluate", "deploy", "monitor"}]
    if not candidates:
        return "- Define next steps\n- Assign owners\n- Set timeline"
    return "\n".join(f"- {c.title()} key measures" for c in candidates[:3])


def extract_conclusion(text: str) -> str:
    """Improved conclusion extraction avoiding decimal splits.

    Prefer the last non-empty line; if unavailable, fall back to sentence
    splitting while skipping fragments that look like numeric tails (e.g.,
    "0" from "0.37").
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    parts = [p for p in parts if not p.isdigit()]
    return parts[-1] if parts else text
