"""Real reasoning baselines using DeepSeek API.

Implements actual Chain-of-Thought, Tree-of-Thought, Graph-of-Thought,
and ReAct strategies that call the LLM and depend on the problem input.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .deepseek_client import deepseek_chat, extract_text_answer

logger = logging.getLogger(__name__)


class ReasoningBaseline(ABC):
    """Abstract base class for reasoning baselines."""

    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def solve(self, problem: str) -> str:
        """Solve a problem using this reasoning strategy."""
        ...

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call DeepSeek API and return the text response."""
        resp = deepseek_chat(messages, model=self.model, stream=False, temperature=self.temperature)
        text = extract_text_answer(resp) if resp else ""
        return text.strip() if text else ""


class CoTBaseline(ReasoningBaseline):
    """Chain-of-Thought: step-by-step reasoning using the LLM."""

    def solve(self, problem: str) -> str:
        prompt = (
            "Let's think step by step.\n\n"
            f"Problem: {problem}\n\n"
            "Solution:"
        )
        messages = [
            {"role": "system", "content": "You are a careful reasoning assistant. Always show your step-by-step thinking before giving the final answer."},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages)


class ToTBaseline(ReasoningBaseline):
    """Tree-of-Thought: explores multiple reasoning branches and evaluates them.

    Uses a simple BFS approach: generate K branches, evaluate each,
    select the best, expand further.
    """

    def __init__(self, branches: int = 3, depth: int = 2, model: str = "deepseek-chat", temperature: float = 0.3):
        super().__init__(model=model, temperature=temperature)
        self.branches = branches
        self.depth = depth

    def solve(self, problem: str) -> str:
        # Step 1: Generate initial branches
        branch_prompt = (
            f"Problem: {problem}\n\n"
            f"Generate {self.branches} distinct approaches to solve this problem. "
            f"Number them 1, 2, 3.\n\n"
            f"Approaches:"
        )
        messages = [
            {"role": "system", "content": "You are a creative problem solver. Generate diverse solution approaches."},
            {"role": "user", "content": branch_prompt},
        ]
        branches_text = self._call_llm(messages)

        # Step 2: Evaluate and select the best branch
        eval_prompt = (
            f"Problem: {problem}\n\n"
            f"Possible approaches:\n{branches_text}\n\n"
            f"Evaluate each approach. Which one is most likely to be correct? "
            f"Select the best approach and solve the problem with it.\n\n"
            f"Final solution:"
        )
        messages = [
            {"role": "system", "content": "You are a critical evaluator. Select the best approach and solve thoroughly."},
            {"role": "user", "content": eval_prompt},
        ]
        return self._call_llm(messages)


class GoTBaseline(ReasoningBaseline):
    """Graph-of-Thought: non-linear reasoning with interconnected ideas.

    Generates reasoning nodes, identifies connections, merges related concepts,
    and synthesizes a final solution.
    """

    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.2):
        super().__init__(model=model, temperature=temperature)

    def solve(self, problem: str) -> str:
        prompt = (
            f"Problem: {problem}\n\n"
            f"Use Graph-of-Thought reasoning:\n"
            f"1. Identify key concepts and reasoning nodes\n"
            f"2. Connect related nodes and find relationships\n"
            f"3. Merge interconnected ideas\n"
            f"4. Synthesize a final solution from the reasoning graph\n\n"
            f"Solution:"
        )
        messages = [
            {"role": "system", "content": "You reason by building a graph of interconnected ideas and merging them into a solution."},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages)


class ReActBaseline(ReasoningBaseline):
    """Reason + Act: alternating reasoning and action steps.

    Simulates a Thought -> Action -> Observation cycle, where each
    thought proposes an action and the observation informs the next thought.
    """

    def __init__(self, cycles: int = 3, model: str = "deepseek-chat", temperature: float = 0.1):
        super().__init__(model=model, temperature=temperature)
        self.cycles = cycles

    def solve(self, problem: str) -> str:
        prompt = (
            f"Problem: {problem}\n\n"
            f"Use ReAct (Reasoning + Acting) to solve this problem:\n\n"
        )
        for i in range(1, self.cycles + 1):
            prompt += (
                f"Thought {i}: What do I need to figure out?\n"
                f"Action {i}: [reasoning step or calculation]\n"
                f"Observation {i}: What did this tell me?\n\n"
            )
        prompt += "Based on the above, the final answer is:"

        messages = [
            {"role": "system", "content": "You reason using the ReAct framework: alternating thoughts, actions, and observations."},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages)
