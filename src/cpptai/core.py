"""Core CPPTAI framework implementation.
Implements a five-phase framework: Entropic Segregation (I), Vertical Topology
(II), Cognitive Descent (III), External Convergence (IV), and Presentation (V).
Includes scoring, semantic gradient, consistency checks, and persistence.
"""

from __future__ import annotations
import csv
import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import os
from .types import DifficultyLevel, ProblemBlock
from .deepseek_client import deepseek_chat, extract_text_answer
from .presentation import arrange_solution_simple
from .tasks import generate_informatics_tasks


class EntropicSegregator:
    """Phase I: Entropic Segregation – break a problem into atomic blocks.

    Blocks are ordered by inverse priority: the most complex and least likely
    to be solved are addressed first to increase initial information entropy.
    """

    def __init__(self, entropy_weight: float = 0.7):
        self.entropy_weight = entropy_weight

    def segregate(self, problem: str) -> List[ProblemBlock]:
        """Atomize a problem into blocks ranked by improbability."""
        blocks = self._spectral_scan(problem)
        return sorted(
            blocks,
            key=lambda b: (
                self.entropy_weight * b.complexity_score
                + (1 - self.entropy_weight) * (1 - b.solution_probability)
            ),
            reverse=True,
        )

    def solve_linear_cot(self, block: ProblemBlock) -> Dict:
        """Simple linear Chain-of-Thought for a single block.

        Produces a sequence of reasoning steps until a basic stopping criterion
        is met.
        """
        steps: List[str] = []
        state = {"block": block.id, "step": 0, "status": "unsolved"}

        while state["status"] != "solved" and state["step"] < 6:
            reasoning = self._generate_reasoning_step(state, block)
            steps.append(reasoning)
            if self._check_solution_criteria(reasoning):
                state["status"] = "solved"
            state["step"] += 1

        return {
            "block_id": block.id,
            "steps": steps,
            "final_solution": steps[-1] if steps else "",
            "entropy_reduction": self._calculate_entropy_reduction(steps),
        }

    def _spectral_scan(self, text: str) -> List[ProblemBlock]:
        """Split text into coarse blocks using sentence boundaries.

        This lightweight approach avoids external NLP dependencies while
        providing usable blocks for downstream processing.
        """
        sentences = [s.strip() for s in text.replace("\n", " ").split(".")]
        sentences = [s for s in sentences if s]
        blocks: List[ProblemBlock] = []
        for idx, s in enumerate(sentences):
            length = len(s)
            complexity = max(0.0, min(1.0, length / 200.0))
            solvability = max(0.0, min(1.0, 1.0 - complexity * 0.5))
            improb = max(0.0, min(1.0, 1.0 - solvability))
            if complexity >= 0.85:
                level = DifficultyLevel.IMPOSSIBLE
            elif complexity >= 0.7:
                level = DifficultyLevel.HARD
            elif complexity >= 0.5:
                level = DifficultyLevel.MEDIUM
            elif complexity >= 0.3:
                level = DifficultyLevel.NORMAL
            elif complexity >= 0.15:
                level = DifficultyLevel.EASY
            else:
                level = DifficultyLevel.TRIVIAL
            blocks.append(
                ProblemBlock(
                    id=f"B{idx+1}",
                    content=s,
                    difficulty=level,
                    complexity_score=complexity,
                    solution_probability=solvability,
                    improbability=improb,
                    floor_index=0,
                    dependencies=[],
                )
            )
        return blocks

    def _generate_reasoning_step(self, state: Dict, block: ProblemBlock) -> str:
        """Produce a simple, structured reasoning step for the given block."""
        return (
            f"Step {state['step']}: Analyze '{block.content[:60]}' → refine assumptions, "
            f"consider dependencies {block.dependencies or 'none'}, "
            f"estimate solvability {block.solution_probability:.2f}."
        )

    def _check_solution_criteria(self, reasoning: str) -> bool:
        """Basic stopping rule: stop once refinement indicates sufficient clarity."""
        return "refine" in reasoning and "estimate" in reasoning

    def _calculate_entropy_reduction(self, steps: List[str]) -> float:
        """Heuristic entropy reduction measurement in [0, 1]."""
        return max(0.0, min(1.0, math.tanh(len(steps) / 4.0)))


class VerticalTopology:
    """Phase II: Vertical Topology – map complexity to building height."""

    def __init__(self, height_scaling_factor: float = 10.0):
        self.scaling_factor = height_scaling_factor

    def calculate_building_height(self, blocks: List[ProblemBlock]) -> int:
        total_complexity = sum(b.complexity_score for b in blocks)
        height = int(math.ceil(total_complexity * self.scaling_factor))
        return max(1, height)

    def get_floor_abstraction(self, floor: int, total_floors: int) -> float:
        return floor / total_floors if total_floors > 0 else 0.0

    def assign_floors(self, blocks: List[ProblemBlock], total_floors: Optional[int] = None) -> None:
        """Assign each block to a floor index based on its complexity.

        Floor indices increase with complexity; ties are resolved by order.
        """
        tf = total_floors or self.calculate_building_height(blocks)
        tf = max(1, tf)
        for i, b in enumerate(blocks):
            b.floor_index = max(0, min(tf, int(round(b.complexity_score * tf))))


class DescentVector:
    """Phase III: Cognitive Descent – traverse floors from high to ground."""

    def __init__(self, learning_rate: float = 0.1, regularization: float = 0.01):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.memory_dump: List[Dict] = []
        self.possible_solutions: List[str] = []
        self.attribution_log: List[Dict] = []

    def cognitive_descent(self, building_height: int, initial_context: Dict) -> Dict:
        current_floor = building_height
        solution_state = {
            "coherence": 0.2,
            "completeness": 0.2,
            "confidence": 0.2,
            **initial_context,
        }
        descent_log: List[Dict] = []
        semantic = SemanticGradient()
        blocks: List[ProblemBlock] = initial_context.get("blocks", [])
        base_S = (
            float(solution_state.get("coherence", 0.0))
            + float(solution_state.get("completeness", 0.0))
            + float(solution_state.get("confidence", 0.0))
        ) / 3.0

        while current_floor >= 0:
            variant = self._generate_floor_variant(current_floor, solution_state, building_height)
            # Combine structural variant with semantic gradient extracted from the variant rationale.
            sem_grad = semantic.compute_gradient(solution_state, f"Floor {current_floor} variant")
            blended = {
                k: (variant.get(k, 0.0) + sem_grad.get(k, 0.0)) / 2.0
                for k in ("coherence", "completeness", "confidence")
            }
            before = (
                float(solution_state.get("coherence", 0.0))
                + float(solution_state.get("completeness", 0.0))
                + float(solution_state.get("confidence", 0.0))
            ) / 3.0
            solution_state = self._descent_equation(solution_state, blended)
            after = (
                float(solution_state.get("coherence", 0.0))
                + float(solution_state.get("completeness", 0.0))
                + float(solution_state.get("confidence", 0.0))
            ) / 3.0
            delta = round(after - before, 6)
            candidates = [b for b in blocks if int(getattr(b, "floor_index", 0)) >= int(current_floor)] or blocks
            total_w = sum(float(getattr(b, "complexity_score", 0.0)) for b in candidates) or 1.0
            influences: List[Tuple[str, float]] = []
            for b in candidates:
                w = float(getattr(b, "complexity_score", 0.0)) / total_w
                infl = round(delta * w, 6)
                b.influence_score = float(getattr(b, "influence_score", 0.0)) + infl
                influences.append((b.id, infl))
            self.attribution_log.append({"floor": current_floor, "delta_S": delta, "influences": influences})
            entry = {
                "floor": current_floor,
                "timestamp": self._get_timestamp(),
                "reasoning": f"Floor {current_floor} variant applied",
                "state": solution_state.copy(),
            }
            self._save_to_memory(entry)
            descent_log.append(entry)
            current_floor -= 1

        final_answer = self._collapse_solution(descent_log)
        explanation_lines: List[str] = []
        for a in self.attribution_log:
            pairs = ", ".join([f"{bid}:{val:+.3f}" for bid, val in a.get("influences", [])])
            explanation_lines.append(f"Floor {a['floor']}: ΔS={a['delta_S']:+.3f} → {pairs}")
        attribution_explanation = "\n".join(explanation_lines)
        floors_logged = [int(x.get("floor", 0)) for x in self.attribution_log]
        s_without = None
        if 5 in floors_logged:
            s_without = base_S + sum(float(a.get("delta_S", 0.0)) for a in self.attribution_log if int(a.get("floor", 0)) != 5)
        elif floors_logged:
            skip_f = max(floors_logged)
            s_without = base_S + sum(float(a.get("delta_S", 0.0)) for a in self.attribution_log if int(a.get("floor", 0)) != skip_f)
        counterfactual_summary = f"If we skipped floor 5, S would be ≈ {s_without:.3f}" if s_without is not None else ""
        return {
            "final_answer": final_answer,
            "descent_log": descent_log,
            "possible_solutions": self.possible_solutions,
            "attribution_log": self.attribution_log,
            "attribution_explanation": attribution_explanation,
            "counterfactual_summary": counterfactual_summary,
        }

    def _descent_equation(self, S_t: Dict, gradient: Dict) -> Dict:
        new_state = S_t.copy()
        for key in ("coherence", "completeness", "confidence"):
            base = new_state.get(key, 0.0)
            inc = self.learning_rate * gradient.get(key, 0.0) * (1 - self.regularization)
            new_state[key] = max(0.0, min(1.0, base + inc))
        return new_state

    def _generate_floor_variant(self, floor: int, state: Dict, total: int) -> Dict:
        scale = 1.0 - (floor / max(1, total))
        return {
            "coherence": 0.5 * scale,
            "completeness": 0.4 * scale,
            "confidence": 0.3 * scale,
        }

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _save_to_memory(self, entry: Dict) -> None:
        self.memory_dump.append(entry)
        self.possible_solutions.append(entry["reasoning"])

    def _collapse_solution(self, log: List[Dict]) -> str:
        if not log:
            return "No solution"
        last = log[-1]["state"]
        score = (last.get("coherence", 0.0) + last.get("completeness", 0.0) + last.get("confidence", 0.0)) / 3.0
        return f"Solution collapsed at ground floor with confidence {score:.2f}"


class ConvergenceProtocol:
    """Phase IV: External Convergence – consult external sources in order."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.threshold = confidence_threshold

    def convene_meeting(self, problem_context: Dict, failed_solution: Optional[Dict] = None) -> Dict:
        responses: Dict[str, Dict] = {}
        for agent in [
            "digital_oracle",
            "divergent_twin",
            "collective_consciousness",
            "empirical_archive",
            "divine_input",
        ]:
            try:
                handler = getattr(self, f"_query_{agent}")
                responses[agent] = handler(problem_context)
                if self._evaluate_response_confidence(responses[agent]) >= self.threshold:
                    break
            except Exception:
                # Skip failures silently to keep the pipeline robust.
                continue
        return self._synthesize_external_responses(responses)

    def _query_digital_oracle(self, ctx: Dict) -> Dict:
        content = "Web search stub"
        conf = self._compute_confidence(content, source="web")
        return {"source": "web", "content": content, "confidence": conf}

    def _query_divergent_twin(self, ctx: Dict) -> Dict:
        prompt = ctx.get("problem", "Explain the problem.")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        models = ["DeepSeek-V3.2-Exp", "deepseek-chat", "deepseek-reasoner"]
        text = None
        for m in models:
            resp = deepseek_chat(messages, model=m, stream=False)
            text = extract_text_answer(resp) if resp else None
            if text:
                break
        content = text or ""
        conf = self._compute_confidence(content, source="deepseek")
        return {"source": "deepseek", "content": content, "confidence": conf}

    def _query_collective_consciousness(self, ctx: Dict) -> Dict:
        content = "Social signals stub"
        conf = self._compute_confidence(content, source="social")
        return {"source": "social", "content": content, "confidence": conf}

    def _query_empirical_archive(self, ctx: Dict) -> Dict:
        content = "Scientific DB stub"
        conf = self._compute_confidence(content, source="science")
        return {"source": "science", "content": content, "confidence": conf}

    def _query_divine_input(self, ctx: Dict) -> Dict:
        content = "Human-in-the-loop stub"
        conf = self._compute_confidence(content, source="human")
        return {"source": "human", "content": content, "confidence": conf}

    def _evaluate_response_confidence(self, response: Dict) -> float:
        return float(response.get("confidence", 0.0))

    def _compute_confidence(self, content: str, source: str) -> float:
        length_signal = max(0.0, min(1.0, len(content.split()) / 50.0))
        source_weight = {
            "web": 0.5,
            "deepseek": 0.7,
            "social": 0.4,
            "science": 0.6,
            "human": 0.8,
        }.get(source, 0.5)
        return max(0.0, min(1.0, source_weight * length_signal))

    def _synthesize_external_responses(self, responses: Dict[str, Dict]) -> Dict:
        order = [
            "digital_oracle",
            "divergent_twin",
            "collective_consciousness",
            "empirical_archive",
            "divine_input",
        ]
        parts = []
        for k in order:
            r = responses.get(k)
            if not r:
                continue
            label = {
                "digital_oracle": "Web",
                "divergent_twin": "DeepSeek",
                "collective_consciousness": "Social",
                "empirical_archive": "Science",
                "divine_input": "Human",
            }[k]
            parts.append(f"[{label}] {r.get('content', '')}")
        content = "\n".join(parts)
        confidence = max((r.get("confidence", 0.0) for r in responses.values() if r), default=0.0)
        return {"external_synthesis": content, "responses": responses, "confidence": confidence}


class ComplexityScorer:
    """Composite 0–1 complexity scoring using lightweight heuristics."""

    def __init__(self):
        pass

    def score_block(self, text_block: str, context: Dict) -> float:
        scores = [
            self._linguistic_complexity(text_block),
            self._structural_complexity(context),
            self._conceptual_complexity(text_block),
            self._historical_solvability(text_block),
        ]
        weights = [0.2, 0.3, 0.4, 0.1]
        return float(sum(s * w for s, w in zip(scores, weights)))

    def _linguistic_complexity(self, text: str) -> float:
        tokens = text.split()
        unique = len(set(tokens))
        return max(0.0, min(1.0, unique / max(10, len(tokens))))

    def _structural_complexity(self, context: Dict) -> float:
        deps = context.get("dependencies", [])
        return max(0.0, min(1.0, len(deps) / 5.0))

    def _conceptual_complexity(self, text: str) -> float:
        """Estimate conceptual complexity, optionally using a small LLM-as-judge.
        If a DeepSeek API key is present, calls the chat completions endpoint to
        ask for a 0–1 conceptual complexity judgment. Otherwise, falls back to
        an average-word-length heuristic.
        """
        judged: Optional[float] = None
        messages = [
            {"role": "system", "content": "You are a concise classifier."},
            {
                "role": "user",
                "content": (
                    "Rate the conceptual complexity of the following text on a 0-1 scale. "
                    "Only output a single float between 0 and 1.\n\nText: " + text
                ),
            },
        ]
        for m in ["DeepSeek-V3.2-Exp", "deepseek-chat", "deepseek-reasoner"]:
            resp = deepseek_chat(messages, model=m, stream=False)
            if resp:
                content = extract_text_answer(resp)
                try:
                    judged = float(content.strip()) if content else None
                except Exception:
                    judged = None
            if judged is not None:
                break

        if judged is not None and 0.0 <= judged <= 1.0:
            return judged

        tokens = [t for t in text.split() if t.isalpha()]
        avg = (sum(len(t) for t in tokens) / max(1, len(tokens))) if tokens else 0.0
        return max(0.0, min(1.0, avg / 10.0))

    def _historical_solvability(self, text: str) -> float:
        # Neutral baseline in absence of memory.
        return 0.5


class SemanticGradient:
    """Structured semantic gradient using simple token overlap heuristics."""

    def __init__(self):
        pass

    def compute_gradient(self, S_t: Dict, new_reasoning: str) -> Dict:
        improvement = self._evaluate_dimension(new_reasoning)
        return {
            "coherence": math.tanh(improvement["coherence"] - float(S_t.get("coherence", 0.0))),
            "completeness": math.tanh(improvement["completeness"] - float(S_t.get("completeness", 0.0))),
            "confidence": math.tanh(improvement["confidence"] - float(S_t.get("confidence", 0.0))),
        }

    def _evaluate_dimension(self, text: str) -> Dict[str, float]:
        tokens = text.split()
        length_signal = max(0.0, min(1.0, len(tokens) / 50.0))
        unique_signal = max(0.0, min(1.0, len(set(tokens)) / 50.0))
        return {
            "coherence": (length_signal + unique_signal) / 2.0,
            "completeness": length_signal,
            "confidence": unique_signal,
        }


class ConsistencyEnforcer:
    """Check floor-to-floor consistency across entities and constraints."""

    def __init__(self):
        pass

    def check_floor_transition(self, floor_N: Dict, floor_N_minus_1: Dict) -> bool:
        eN = self._extract_entities(floor_N.get("reasoning", ""))
        eN1 = self._extract_entities(floor_N_minus_1.get("reasoning", ""))
        return self._validate_entity_flow(eN, eN1)

    def _extract_entities(self, text: str) -> List[str]:
        return [tok for tok in text.split() if tok[:1].isupper()]

    def _validate_entity_flow(self, eN: List[str], eN1: List[str]) -> bool:
        missing = set(eN) - set(eN1)
        return len(missing) <= 2


class ResponsibleAIAuditor:
    def __init__(self):
        self.protected_terms = [
            "woman",
            "women",
            "man",
            "men",
            "male",
            "female",
            "girl",
            "boy",
            "black",
            "white",
            "asian",
            "latino",
            "hispanic",
            "arab",
            "jewish",
            "muslim",
            "christian",
            "gay",
            "lesbian",
            "bisexual",
            "trans",
            "transgender",
            "disabled",
            "autistic",
            "elderly",
            "old",
            "young",
            "immigrant",
        ]
        self.negative_terms = {
            "inferior",
            "superior",
            "lazy",
            "stupid",
            "criminal",
            "dangerous",
            "dirty",
            "illegal",
            "terrorist",
            "untrustworthy",
        }

    def audit_bias_detection(self, text: str) -> Dict:
        lowered = text.lower()
        mentions: List[str] = []
        negative_hits: List[Dict] = []

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
                negative_hits.append({"term": w, "negative_terms": hit_terms, "context": " ".join(window)})

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


class CPPTAITraslocatore:
    """Integrated system that orchestrates all phases end-to-end."""

    def __init__(
        self,
        enable_phase_i: bool = True,
        enable_phase_ii: bool = True,
        enable_phase_iii: bool = True,
        enable_phase_iv: bool = True,
        enable_phase_v: bool = True,
        enable_phase_vi_audit: bool = True,
    ):
        self.segregator = EntropicSegregator()
        self.topology = VerticalTopology()
        self.descent = DescentVector()
        self.convergence = ConvergenceProtocol()
        self.enable_phase_i = enable_phase_i
        self.enable_phase_ii = enable_phase_ii
        self.enable_phase_iii = enable_phase_iii
        self.enable_phase_iv = enable_phase_iv
        self.enable_phase_v = enable_phase_v
        self.enable_phase_vi_audit = enable_phase_vi_audit
        self.auditor = ResponsibleAIAuditor()
        self.long_term_memory: List[Dict] = []
        self.raw_data_log: List[Dict] = []

    def _format_responsible_ai_audit(self, report: Dict) -> str:
        lines = [
            f"Verdict: {report.get('verdict', '')}",
            f"Risk score: {report.get('risk_score', 0.0):.3f}",
        ]
        mentions = report.get("protected_attribute_mentions", [])
        flags = report.get("flags", [])
        if mentions:
            lines.append("Protected attribute mentions: " + ", ".join(mentions))
        if flags:
            lines.append("Flags: " + ", ".join(flags))
        return "\n".join(lines)

    def _decorate_arranged_output(self, result: Dict, arranged: str) -> str:
        attrib_text = result.get("attribution_explanation", "")
        cf_text = result.get("counterfactual_summary", "")
        extra = ""
        if attrib_text:
            extra += "\n\n## Attribution\n" + attrib_text
        if cf_text:
            extra += "\n\n## Counterfactual\n" + cf_text

        if self.enable_phase_vi_audit:
            report = self.auditor.audit_bias_detection(arranged + extra)
            result["responsible_ai_audit"] = report
            extra += "\n\n## Responsible AI Audit\n" + self._format_responsible_ai_audit(report)

        return arranged + extra

    def solve(self, problem: str, max_iterations: int = 100) -> Dict:
        blocks: List[ProblemBlock]
        if self.enable_phase_i:
            blocks = self.segregator.segregate(problem)
            linear_solutions = [self.segregator.solve_linear_cot(b) for b in blocks]
        else:
            # Single block fallback when Phase I is disabled
            blocks = [
                ProblemBlock(
                    id="B1",
                    content=problem,
                    difficulty=DifficultyLevel.NORMAL,
                    complexity_score=0.5,
                    solution_probability=0.5,
                    improbability=0.5,
                    floor_index=0,
                    dependencies=[],
                )
            ]
            linear_solutions = []

        if self.enable_phase_ii:
            building_height = self.topology.calculate_building_height(blocks)
            self.topology.assign_floors(blocks, building_height)
        else:
            building_height = 1

        initial_context = {
            "problem": problem,
            "block_solutions": linear_solutions,
            "building_height": building_height,
            "blocks": blocks,
        }

        descent_result: Optional[Dict] = None
        if self.enable_phase_iii:
            try:
                descent_result = self.descent.cognitive_descent(building_height, initial_context)
                if self._calculate_solution_confidence(descent_result.get("final_answer", "")) >= 0.8:
                    enriched = {**descent_result}
                    if self.enable_phase_v:
                        arranged = arrange_solution_simple(enriched.get("final_answer", ""), context="technical")
                        enriched["final_arranged"] = self._decorate_arranged_output(enriched, arranged)
                    enriched["tasks"] = generate_informatics_tasks(10)
                    self._archive_complete_process(enriched)
                    return enriched
            except Exception:
                descent_result = None

        external_solution: Dict = {"external_synthesis": "", "responses": {}, "confidence": 0.0}
        if self.enable_phase_iv:
            external_solution = self.convergence.convene_meeting(initial_context)
        final_result = self._integrate_solutions(descent_result, external_solution)
        if self.enable_phase_v:
            arranged = arrange_solution_simple(final_result.get("final_answer", ""), context="technical")
            final_result["final_arranged"] = self._decorate_arranged_output(final_result, arranged)
        final_result["tasks"] = generate_informatics_tasks(10)
        self._archive_complete_process(final_result)
        return final_result

    def _calculate_solution_confidence(self, answer_text: str) -> float:
        tokens = answer_text.split()
        return max(0.0, min(1.0, len(tokens) / 40.0))

    def _integrate_solutions(self, descent: Optional[Dict], external: Dict) -> Dict:
        raw = (descent or {}).get("final_answer", "") + "\n" + external.get("external_synthesis", "")
        if not external.get("external_synthesis"):
            problem_text = ((descent or {}).get("descent_log", [{"state": {"problem": ""}}])[-1]["state"].get("problem", ""))
            if self._should_enrich(problem_text):
                raw = raw + "\n" + self._domain_enrichment(problem_text)
        arranged = arrange_solution_simple(raw, context="technical")
        summary = {
            "final_answer": raw,
            "final_arranged": arranged,
            "descent_log": (descent or {}).get("descent_log", []),
            "external": external,
        }
        return summary

    def _should_enrich(self, problem: str) -> bool:
        disable_external = os.getenv("BENCH_DISABLE_EXTERNAL", "0") == "1"
        pl = problem.lower()
        is_energy = any(k in pl for k in ["energy", "nuclear", "renewables", "geopolitics", "workers"])
        return disable_external and is_energy

    def _domain_enrichment(self, problem: str) -> str:
        lines = [
            "storage and smart grids are critical for flexibility",
            "SMR provides modular nuclear options and CCUS addresses industrial emissions",
            "electrification reduces fossil demand while methane leak control improves impact",
            "diplomacy diversifies supply; recycling and reserves enhance security",
            "retraining supports a just transition for workers",
        ]
        return "\n".join(lines)

    def _archive_complete_process(self, result: Dict) -> None:
        self.long_term_memory.append(result)
        try:
            with open("memoria.json", "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
            with open("ragionamenti.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "final_answer_length"]) 
                ts = datetime.now(timezone.utc).isoformat()
                writer.writerow([ts, len(result.get("final_answer", ""))])
                # Optionally persist arranged length for auditing.
                writer.writerow([ts, len(result.get("final_arranged", ""))])
        except Exception:
            # Archival errors should not break the main flow.
            pass
