"""Core CPPTAI framework implementation.
This module implements the four-phase architecture described in the provided
document: Entropic Segregation, Vertical Topology, Cognitive Descent, and
External Convergence. It also includes scoring, semantic gradient, and
consistency checks. All code and comments are in English.
"""

from __future__ import annotations
import csv
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
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


class DescentVector:
    """Phase III: Cognitive Descent – traverse floors from high to ground."""

    def __init__(self, learning_rate: float = 0.1, regularization: float = 0.01):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.memory_dump: List[Dict] = []
        self.possible_solutions: List[str] = []

    def cognitive_descent(self, building_height: int, initial_context: Dict) -> Dict:
        current_floor = building_height
        solution_state = {
            "coherence": 0.2,
            "completeness": 0.2,
            "confidence": 0.2,
            **initial_context,
        }
        descent_log: List[Dict] = []

        while current_floor >= 0:
            variant = self._generate_floor_variant(current_floor, solution_state, building_height)
            solution_state = self._descent_equation(solution_state, variant)
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
        return {
            "final_answer": final_answer,
            "descent_log": descent_log,
            "possible_solutions": self.possible_solutions,
        }

    def _descent_equation(self, S_t: Dict, gradient: Dict) -> Dict:
        new_state = S_t.copy()
        for key in ("coherence", "completeness", "confidence"):
            new_state[key] = max(0.0, min(1.0, new_state.get(key, 0.0) + self.learning_rate * gradient.get(key, 0.0)))
        for key in ("coherence", "completeness", "confidence"):
            new_state[key] *= (1 - self.regularization)
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
        return {"source": "web", "content": "Web search stub", "confidence": 0.4}

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
        return {"source": "deepseek", "content": text or "", "confidence": 0.6 if text else 0.2}

    def _query_collective_consciousness(self, ctx: Dict) -> Dict:
        return {"source": "social", "content": "Social signals stub", "confidence": 0.3}

    def _query_empirical_archive(self, ctx: Dict) -> Dict:
        return {"source": "science", "content": "Scientific DB stub", "confidence": 0.5}

    def _query_divine_input(self, ctx: Dict) -> Dict:
        return {"source": "human", "content": "Human-in-the-loop stub", "confidence": 0.8}

    def _evaluate_response_confidence(self, response: Dict) -> float:
        return float(response.get("confidence", 0.0))

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


class CPPTAITraslocatore:
    """Integrated system that orchestrates all phases end-to-end."""

    def __init__(self):
        self.segregator = EntropicSegregator()
        self.topology = VerticalTopology()
        self.descent = DescentVector()
        self.convergence = ConvergenceProtocol()
        self.long_term_memory: List[Dict] = []
        self.raw_data_log: List[Dict] = []

    def solve(self, problem: str, max_iterations: int = 100) -> Dict:
        blocks = self.segregator.segregate(problem)
        linear_solutions = [self.segregator.solve_linear_cot(b) for b in blocks]
        building_height = self.topology.calculate_building_height(blocks)
        initial_context = {
            "problem": problem,
            "block_solutions": linear_solutions,
            "building_height": building_height,
        }
        try:
            descent_result = self.descent.cognitive_descent(building_height, initial_context)
            if self._calculate_solution_confidence(descent_result.get("final_answer", "")) >= 0.8:
                enriched = {**descent_result, "tasks": generate_informatics_tasks(10)}
                self._archive_complete_process(enriched)
                return enriched
        except Exception:
            pass
        external_solution = self.convergence.convene_meeting(initial_context)
        final_result = self._integrate_solutions(descent_result if "descent_result" in locals() else None, external_solution)
        final_result["tasks"] = generate_informatics_tasks(10)
        self._archive_complete_process(final_result)
        return final_result

    def _calculate_solution_confidence(self, answer_text: str) -> float:
        tokens = answer_text.split()
        return max(0.0, min(1.0, len(tokens) / 40.0))

    def _integrate_solutions(self, descent: Optional[Dict], external: Dict) -> Dict:
        raw = (descent or {}).get("final_answer", "") + "\n" + external.get("external_synthesis", "")
        arranged = arrange_solution_simple(raw, context="technical")
        summary = {
            "final_answer": raw,
            "final_arranged": arranged,
            "descent_log": (descent or {}).get("descent_log", []),
            "external": external,
        }
        return summary

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
