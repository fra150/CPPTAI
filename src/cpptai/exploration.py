"""Explorer and Analyzer phases — diffusion for reasoning.

The ExplorerEngine generates multiple interpretations (trajectories) from
a single problem by injecting noise and iteratively denoising. The
TrajectoryAnalyzer selects the best trajectories and produces an enriched
problem statement for downstream phases.
"""

from __future__ import annotations

import os
import random
import logging
import functools
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from .types import ExplorationTrajectory, AnalyzerPrepared
from .deepseek_client import deepseek_chat, extract_text_answer


logger = logging.getLogger(__name__)


EXPLORER_LENSES = [
    "OPTIMIZATION: {problem} — What is the optimal configuration?",
    "CONSTRAINT: {problem} — What are the binding constraints?",
    "TRADEOFF: {problem} — What must be sacrificed?",
    "SYSTEM: {problem} — How do components interact?",
    "UNCERTAINTY: {problem} — What is unknown?",
    "ETHICS: {problem} — Who is affected and how?",
    "TIMELINE: {problem} — What is the urgency?",
    "RESOURCE: {problem} — What is scarce?",
    "CAUSALITY: {problem} — What is the root cause?",
    "PREDICTION: {problem} — What will happen next?",
    "SCENARIO: {problem} — What if conditions change?",
    "RISK: {problem} — What could go wrong?",
    "OPPORTUNITY: {problem} — What upside exists?",
    "STRATEGY: {problem} — What is the best path forward?",
    "INNOVATION: {problem} — What novel approach solves this?",
    "COMPETITION: {problem} — Who else is solving this?",
]


class ExplorerEngine:
    """Implementation of "diffusion for reasoning" — generates multiple
    interpretations from uncertainty.

    The engine injects noise into the original problem (via semantic lenses),
    then iteratively denoises each trajectory over several steps. The result
    is a diverse set of framings ranked by confidence and novelty.

    Supports adaptive trajectory counts, concurrent denoising, batched
    multi-problem exploration, and result caching.
    """

    def __init__(
        self,
        num_trajectories: int = 5,
        noise_level: float = 0.6,
        temperature: float = 0.8,
        denoising_steps: int = 3,
        seed: int = 0,
        adaptive: bool = False,
        max_workers: int = 4,
        parallel_llm: bool = False,
        cache_results: bool = True,
        diversity_weight: float = 0.6,
        offline_mode: bool = False,
    ) -> None:
        """Initialize the ExplorerEngine.

        Args:
            num_trajectories: Number of trajectories to generate.
            noise_level: Proportion of injected noise — controls how far from
                the original the initial framings may deviate.
            temperature: Sampling temperature for LLM-based denoising.
            denoising_steps: Number of iterative denoising passes per trajectory.
            seed: Random seed for reproducible lens selection and RNG state.
            adaptive: When True, auto-scale trajectory count based on problem length.
            max_workers: Maximum threads for concurrent trajectory generation.
            parallel_llm: When True, run LLM denoising calls in parallel (may hit rate limits).
            cache_results: When True, cache explorer results keyed by (problem, seed, num_trajectories).
            diversity_weight: Weight given to novelty in scoring (0-1).
        """
        self.num_trajectories = num_trajectories
        self.noise_level = max(0.0, min(1.0, noise_level))
        self.temperature = max(0.0, min(1.0, temperature))
        self.denoising_steps = max(1, denoising_steps)
        self.seed = seed
        self.adaptive = adaptive
        self.max_workers = max_workers
        self.parallel_llm = parallel_llm
        self.cache_results = cache_results
        self.diversity_weight = max(0.0, min(1.0, diversity_weight))
        self.offline_mode = offline_mode

        self._cache: Dict[str, List[ExplorationTrajectory]] = {}

    def _cache_key(self, problem: str, seed: int, num_trajectories: int) -> str:
        h = hashlib.sha256(f"{problem}::{seed}::{num_trajectories}".encode()).hexdigest()
        return f"explore:{h}"

    def _compute_optimal_trajectories(self, problem: str) -> int:
        word_count = len(problem.split())
        if word_count < 50:
            n = 3
        elif word_count <= 200:
            n = 8
        else:
            n = 15
        return min(n, self.num_trajectories)

    def explore(self, problem: str) -> List[ExplorationTrajectory]:
        """Main entry point — generate a diverse set of trajectories.

        For each trajectory, injects noise via a randomly selected lens,
        then iteratively denoises over ``denoising_steps``. After all
        trajectories are generated, each one is scored for confidence
        and novelty. The final list is sorted by confidence descending.

        Args:
            problem: The original problem statement to explore.

        Returns:
            A list of ExplorationTrajectory objects, sorted by confidence
            descending.
        """
        effective_n = self._compute_optimal_trajectories(problem) if self.adaptive else self.num_trajectories

        if self.cache_results:
            key = self._cache_key(problem, self.seed, effective_n)
            if key in self._cache:
                logger.info("Cache hit for problem (seed=%d, n=%d)", self.seed, effective_n)
                return self._cache[key]

        trajectories: List[ExplorationTrajectory] = []

        if self.parallel_llm:
            trajectories = self._explore_concurrent(problem, effective_n)
        else:
            trajectories = self._explore_sequential(problem, effective_n)

        for traj in trajectories:
            traj.confidence = self._score_trajectory(
                traj.interpretation, traj.reasoning_path, traj.id, trajectories
            )

        self._compute_novelty(trajectories)

        trajectories.sort(key=lambda t: t.confidence, reverse=True)

        if self.cache_results:
            self._cache[key] = trajectories

        return trajectories

    def _explore_sequential(self, problem: str, n: int) -> List[ExplorationTrajectory]:
        trajectories: List[ExplorationTrajectory] = []
        for i in range(n):
            traj = self._generate_single_trajectory(problem, i)
            trajectories.append(traj)
        return trajectories

    def _explore_concurrent(self, problem: str, n: int) -> List[ExplorationTrajectory]:
        trajectories: List[ExplorationTrajectory] = [None] * n
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate_single_trajectory, problem, i): i
                for i in range(n)
            }
            for future in as_completed(futures):
                idx = futures[future]
                trajectories[idx] = future.result()
        return trajectories

    def _generate_single_trajectory(self, problem: str, index: int) -> ExplorationTrajectory:
        rng = random.Random(self.seed + index * 7 + 13)
        seed = self.seed + index * 31

        noise_span = self.noise_level * 1.5 - self.noise_level * 0.5
        trajectory_noise = self.noise_level * 0.5 + (index / max(1, self.num_trajectories - 1)) * noise_span
        trajectory_noise = max(0.0, min(1.0, trajectory_noise))

        noisy = self._inject_noise(problem, seed)
        reasoning: List[str] = []
        current = noisy

        for step in range(self.denoising_steps):
            current = self._denoise_step(current, problem, step, self.denoising_steps, rng)
            reasoning.append(current)

        traj = ExplorationTrajectory(
            id=f"T{index + 1}",
            interpretation=current,
            confidence=0.0,
            reasoning_path=reasoning,
            noise_seed=seed,
        )
        traj.metadata["noise_level"] = trajectory_noise
        return traj

    def explore_batch(self, problems: List[str]) -> List[List[ExplorationTrajectory]]:
        """Process multiple problems in parallel.

        Args:
            problems: List of problem statements.

        Returns:
            A list of trajectory lists, one per problem, in the same order as input.
        """
        results: List[List[ExplorationTrajectory]] = [None] * len(problems)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.explore, problems[i]): i
                for i in range(len(problems))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def clear_cache(self) -> None:
        self._cache.clear()

    def select_best(self, trajectories: List[ExplorationTrajectory]) -> ExplorationTrajectory:
        """Return the trajectory with the highest confidence score.

        Args:
            trajectories: List of trajectories to evaluate.

        Returns:
            The highest-confidence trajectory. If the list is empty, returns
            a default trajectory with a descriptive fallback message.
        """
        if not trajectories:
            return ExplorationTrajectory(
                id="T0",
                interpretation="No trajectories available.",
                confidence=0.0,
                reasoning_path=[],
                noise_seed=0,
            )
        return max(trajectories, key=lambda t: t.confidence)

    def _step_temperature(self, step: int) -> float:
        """Compute a decaying temperature for the given denoising step.

        Later steps use lower temperature to favour more focused refinements.

        Args:
            step: Current step index (0-based).

        Returns:
            A temperature value in [0.1, self.temperature].
        """
        decay = 1.0 - (step / max(1, self.denoising_steps)) * 0.5
        return max(0.1, self.temperature * decay)

    def _inject_noise(self, problem: str, seed: int) -> str:
        """Create an initial noise-injected framing using a semantic lens.

        Selects one lens from EXPLORER_LENSES deterministically from the
        provided seed, formats it with the problem text, and returns the
        resulting interpretation string.

        Args:
            problem: The original problem statement.
            seed: Random seed for deterministic lens selection.

        Returns:
            A formatted interpretation string with a lens prefix.
        """
        rng = random.Random(seed)
        lens_template = rng.choice(EXPLORER_LENSES)
        return lens_template.format(problem=problem)

    def _denoise_step(
        self,
        current: str,
        original: str,
        step: int,
        total_steps: int,
        rng: random.Random,
    ) -> str:
        """Perform one denoising step on the current interpretation.

        If DEEPSEEK_API_KEY is set in the environment, uses the LLM via
        ``deepseek_chat`` with a temperature that decreases per step.
        Otherwise falls back to the heuristic denoiser.

        Args:
            current: The current interpretation string to refine.
            original: The original problem text for context.
            step: Current denoising step index (0-based).
            total_steps: Total number of denoising steps planned.
            rng: Random instance for reproducibility in fallback.

        Returns:
            The refined interpretation after this denoising step.
        """
        if self.offline_mode:
            return self._heuristic_denoise(current, original, step, rng)
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            return self._heuristic_denoise(current, original, step, rng)

        temperature = self._step_temperature(step)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning optimizer. Your task is to refine "
                    "a noisy interpretation of a problem into a clearer, more "
                    "precise framing. Remove ambiguity and sharpen the focus."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original: {original[:300]}\n"
                    f"Current: {current[:300]}\n"
                    f"Refine this interpretation:"
                ),
            },
        ]
        resp = deepseek_chat(
            messages, model="deepseek-chat", stream=False, temperature=temperature
        )
        text = extract_text_answer(resp) if resp else None
        if text and text.strip():
            return text.strip()
        return self._heuristic_denoise(current, original, step, rng)

    def _heuristic_denoise(
        self,
        current: str,
        original: str,
        step: int,
        rng: random.Random,
    ) -> str:
        """Offline heuristic denoising — no LLM required.

        Extracts key terms (words longer than 4 characters) from the original
        problem and returns a progressively more focused template as the step
        index increases.

        Args:
            current: The current interpretation (ignored in this fallback).
            original: The original problem text for keyword extraction.
            step: Current denoising step index.
            rng: Random instance for deterministic template selection.

        Returns:
            A denoised interpretation string built from templates.
        """
        words = [w.strip(".,!?;:()[]{}") for w in original.split()]
        keywords = sorted(set(w for w in words if len(w) > 4))

        step_ratio = max(0.0, min(1.0, step / max(1, self.denoising_steps)))

        templates = [
            (
                f"CRITICAL ANALYSIS: The core of this problem involves "
                f"{' and '.join(keywords[:3]) or 'multiple factors'}. "
                f"A rigorous approach must address these dimensions systematically."
            ),
            (
                f"STRUCTURED FRAMING: Considering "
                f"{' and '.join(keywords[:3]) or 'all aspects'}, "
                f"the problem reduces to a set of interconnected sub-problems "
                f"that can be tackled sequentially."
            ),
            (
                f"BOTTLENECK VIEW: Among the key elements "
                f"({' and '.join(keywords[:4]) or 'the main constraints'}), "
                f"the primary bottleneck determines the overall feasibility."
            ),
            (
                f"SYSTEMIC PERSPECTIVE: The problem space spans "
                f"{' and '.join(keywords[:3]) or 'multiple domains'}. "
                f"Trade-offs between these dimensions drive the solution space."
            ),
        ]

        idx = min(len(templates) - 1, int(step_ratio * len(templates)))
        return templates[idx]

    def _score_trajectory(
        self,
        interpretation: str,
        reasoning_path: List[str],
        trajectory_id: str,
        all_trajectories: List[ExplorationTrajectory],
    ) -> float:
        """Compute a confidence score in [0, 1] for a single trajectory.

        The score is a weighted combination of four signals:

        - **Length** (30%): longer interpretations tend to have more substance.
        - **Reasoning depth** (30%): more denoising steps indicate deeper refinement.
        - **Uniqueness** (20% * dw): semantic distance from other trajectories.
        - **Keyword diversity** (20% * dw): vocabulary richness within the interpretation.

        All sub-scores are clamped to [0, 1] before aggregation.

        Args:
            interpretation: The final interpretation string.
            reasoning_path: The list of intermediate reasoning steps.
            trajectory_id: Unique identifier for this trajectory.
            all_trajectories: All generated trajectories for context.

        Returns:
            A confidence float in [0, 1].
        """
        dw = self.diversity_weight
        interpretation_words = interpretation.split()
        length_score = max(0.0, min(1.0, len(interpretation_words) / 40.0))

        depth_score = max(
            0.0, min(1.0, len(reasoning_path) / max(1, self.denoising_steps))
        )

        unique_score = 0.5
        others = [t for t in all_trajectories if t.id != trajectory_id]
        if others:
            my_keywords = self._keyword_set(interpretation_words)
            overlaps: List[float] = []
            for other in others:
                other_keywords = self._keyword_set(other.interpretation.split())
                if not my_keywords and not other_keywords:
                    overlaps.append(1.0)
                elif not my_keywords or not other_keywords:
                    overlaps.append(0.0)
                else:
                    jaccard = len(my_keywords & other_keywords) / len(my_keywords | other_keywords)
                    overlaps.append(jaccard)
            unique_score = 1.0 - (sum(overlaps) / len(overlaps))

        unique_keywords = len(
            set(w.lower().strip(".,!?;:()[]{}") for w in interpretation_words)
        )
        diversity_score = max(
            0.0, min(1.0, unique_keywords / max(1, len(interpretation_words)))
        )

        score = (
            0.30 * length_score
            + 0.30 * depth_score
            + (0.20 * dw) * unique_score
            + (0.20 * dw) * diversity_score
        )
        return max(0.0, min(1.0, score))

    def _keyword_set(self, words: List[str]) -> Set[str]:
        """Extract a lower-case keyword set from a list of tokens.

        Filters out short tokens (length <= 2) and strips common punctuation.

        Args:
            words: A list of raw word tokens.

        Returns:
            A set of normalised keyword strings.
        """
        return set(
            w.lower().strip(".,!?;:()[]{}\"'")
            for w in words
            if len(w.strip(".,!?;:()[]{}\"'")) > 2
        )

    def _compute_novelty(self, trajectories: List[ExplorationTrajectory]) -> None:
        """Compute and assign novelty scores for all trajectories in-place.

        Novelty for a trajectory is defined as::

            novelty = 1 - max_j Jaccard(keywords_i, keywords_j)

        where the maximum is taken over all other trajectories ``j``.
        Empty keyword sets yield a novelty of 0.0. A singleton list
        receives novelty 1.0.

        Args:
            trajectories: The list of trajectories to score. Each trajectory's
                ``novelty_score`` attribute is updated directly.
        """
        n = len(trajectories)
        if n == 0:
            return
        if n == 1:
            trajectories[0].novelty_score = 1.0
            return

        keyword_sets: List[Set[str]] = [
            self._keyword_set(t.interpretation.split()) for t in trajectories
        ]

        for i, traj in enumerate(trajectories):
            max_sim = 0.0
            mine = keyword_sets[i]
            for j, other_set in enumerate(keyword_sets):
                if i == j:
                    continue
                if not mine and not other_set:
                    sim = 1.0
                elif not mine or not other_set:
                    sim = 0.0
                else:
                    sim = len(mine & other_set) / len(mine | other_set)
                max_sim = max(max_sim, sim)
            traj.novelty_score = max(0.0, min(1.0, 1.0 - max_sim))


class TrajectoryAnalyzer:
    """Select and enrich the best trajectories from ExplorerEngine.

    Takes raw trajectories, ranks by confidence, applies a coherence
    threshold, keeps the top-K ensemble members, and produces an
    AnalyzerPrepared structure with an enriched problem statement
    that is ready for the downstream CPPTAI phases.
    """

    def __init__(
        self,
        ensemble_size: int = 3,
        coherence_threshold: float = 0.5,
    ) -> None:
        """Initialize the TrajectoryAnalyzer.

        Args:
            ensemble_size: Maximum number of top trajectories to retain.
            coherence_threshold: Minimum confidence required for a trajectory
                to be included in the ensemble.
        """
        self.ensemble_size = max(1, ensemble_size)
        self.coherence_threshold = max(0.0, min(1.0, coherence_threshold))

    def analyze(
        self,
        trajectories: List[ExplorationTrajectory],
        original_problem: str,
    ) -> AnalyzerPrepared:
        """Main entry point — select, enrich, and consolidate trajectories.

        Steps:
            1. Sort trajectories by descending confidence.
            2. Filter out those below ``coherence_threshold``.
            3. Keep the top ``ensemble_size`` from the filtered set.
            4. Build an enriched problem by combining the original with
               selected interpretations.
            5. Compute aggregate confidence and identify patterns.

        Gracefully handles the empty-trajectories edge case by returning
        the original problem as-is.

        Args:
            trajectories: Raw trajectories from ExplorerEngine.
            original_problem: The original problem statement.

        Returns:
            An AnalyzerPrepared instance ready for Phase I.
        """
        if not trajectories:
            return AnalyzerPrepared(
                enriched_problem=original_problem,
                top_interpretations=[],
                aggregate_confidence=0.0,
                reasoning_summary="No trajectories were generated.",
                trajectories_used=0,
                patterns_identified=[],
                metadata={
                    "note": "empty trajectories — returned original problem as-is"
                },
            )

        sorted_traj = sorted(trajectories, key=lambda t: t.confidence, reverse=True)

        filtered = [t for t in sorted_traj if t.confidence >= self.coherence_threshold]

        if not filtered:
            filtered = sorted_traj[:1]

        selected = filtered[: self.ensemble_size]

        top_interpretations = [t.interpretation for t in selected]
        aggregate_confidence = sum(t.confidence for t in selected) / len(selected)

        enriched = self._build_enriched_problem(original_problem, top_interpretations)
        summary = self._build_reasoning_summary(selected, original_problem)
        patterns = self._identify_patterns(trajectories)

        return AnalyzerPrepared(
            enriched_problem=enriched,
            top_interpretations=top_interpretations,
            aggregate_confidence=max(0.0, min(1.0, aggregate_confidence)),
            reasoning_summary=summary,
            trajectories_used=len(selected),
            patterns_identified=patterns,
        )

    def _build_enriched_problem(
        self,
        original: str,
        interpretations: List[str],
    ) -> str:
        """Combine the original problem with selected interpretations.

        Produces a structured markdown string with ``##``-level section
        headers and an ``Analysis Directive`` block to guide downstream
        processing phases.

        Args:
            original: The original problem statement.
            interpretations: The top interpretation strings.

        Returns:
            An enriched problem string with sections and directive.
        """
        parts: List[str] = [
            f"## Original Problem\n\n{original}",
            "",
            "## Enriched Framings\n",
        ]
        for i, interp in enumerate(interpretations, 1):
            parts.append(f"### Framing {i}\n\n{interp}")
            parts.append("")

        parts.append("## Analysis Directive\n")
        parts.append(
            "Consider all the above framings when solving. "
            "Address each framing's core question and synthesize "
            "a unified response that accounts for multiple perspectives."
        )

        return "\n".join(parts).strip()

    def _build_reasoning_summary(
        self,
        selected: List[ExplorationTrajectory],
        original: str,
    ) -> str:
        """Create a human-readable summary of the analysis results.

        Lists each selected trajectory with its confidence and novelty
        scores, then reports the aggregate confidence.

        Args:
            selected: The selected trajectories after filtering.
            original: The original problem statement.

        Returns:
            A multi-line text summary.
        """
        if not selected:
            return "No trajectories were selected for analysis."

        lines: List[str] = [
            f"Analyzed {len(selected)} trajectory/ies for: {original[:100]}",
            "",
        ]
        for i, traj in enumerate(selected, 1):
            lines.append(
                f"  {i}. {traj.interpretation[:120]} "
                f"(confidence={traj.confidence:.3f}, "
                f"novelty={traj.novelty_score:.3f})"
            )

        avg_conf = sum(t.confidence for t in selected) / len(selected)
        lines.append("")
        lines.append(f"Aggregate confidence: {avg_conf:.3f}")

        return "\n".join(lines)

    def _identify_patterns(
        self,
        trajectories: List[ExplorationTrajectory],
    ) -> List[str]:
        """Identify common keywords across all trajectories as "patterns".

        A keyword (word longer than 4 characters) is considered a pattern
        when it appears in more than half of the trajectories. Results are
        sorted by frequency descending and capped at 10 entries.

        Args:
            trajectories: All generated trajectories (including those below
                the coherence threshold).

        Returns:
            A list of pattern strings, most frequent first.
        """
        if not trajectories:
            return []

        trajectory_keywords: List[Set[str]] = []
        for traj in trajectories:
            words = traj.interpretation.split()
            keywords = set(
                w.lower().strip(".,!?;:()[]{}") for w in words if len(w) > 4
            )
            trajectory_keywords.append(keywords)

        word_counts: Dict[str, int] = {}
        for kw_set in trajectory_keywords:
            for kw in kw_set:
                word_counts[kw] = word_counts.get(kw, 0) + 1

        threshold = max(1, len(trajectories) // 2)
        common = sorted(
            [w for w, c in word_counts.items() if c >= threshold],
            key=lambda w: word_counts[w],
            reverse=True,
        )

        return common[:10]
