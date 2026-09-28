"""Dataset loaders for broader benchmarking (GSM8K, MATH, HumanEval, SciBench).
Currently provides stubs and synthetic generators, as actual datasets require external files.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
import random

try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

logger = logging.getLogger(__name__)


def _categorize_gsm8k(question: str) -> str:
    """Categorize a GSM8K problem by type.

    Returns one of:
    - 'one_step': single arithmetic operation
    - 'multi_step': multiple operations
    - 'word_problem': story-based with multiple entities
    - 'comparison': comparing quantities
    - 'fraction': involving fractions or percentages
    """
    q = question.lower()
    if any(w in q for w in ["how many more", "how many fewer", "difference", "compare"]):
        return "comparison"
    if any(w in q for w in ["fraction", "percent", "%", "half", "quarter"]):
        return "fraction"
    if any(w in q for w in ["total", "altogether", "combined", "sum", "together"]):
        word_count = len(q.split())
        if word_count < 30:
            return "one_step"
        return "multi_step"
    word_count = len(q.split())
    if word_count < 30:
        return "one_step"
    return "word_problem"


class DatasetLoader:
    @staticmethod
    def load_gsm8k(n: int = 1319, force_stub: bool = False) -> List[Dict]:
        """Loads GSM8K (Grade School Math 8K) problems — ALL 1319 test problems.

        Args:
            n: Max problems to load. Default 1319 = full test set.
            force_stub: If True, skip HuggingFace and use synthetic data.

        Returns:
            List of problems with id, prompt, expected, dataset.
        """
        problems = []
        if not force_stub and HAS_HF_DATASETS:
            try:
                ds = load_dataset("gsm8k", "main", split="test", streaming=True)
                iterator = iter(ds)
                for i in range(n):
                    try:
                        item = next(iterator)
                        answer = item["answer"]
                        # Extract the numeric answer after "####"
                        if "####" in answer:
                            numeric_answer = answer.split("####")[-1].strip()
                        else:
                            numeric_answer = answer.strip()
                        problems.append({
                            "id": f"gsm8k_{i+1}",
                            "prompt": item["question"],
                            "expected": [numeric_answer],
                            "dataset": "gsm8k",
                            "metadata": {
                                "full_answer": answer,
                                "category": _categorize_gsm8k(item["question"]),
                            }
                        })
                    except StopIteration:
                        break
                print(f"GSM8K: loaded {len(problems)} real problems from HuggingFace")
                return problems
            except Exception as e:
                print(f"GSM8K HF load failed: {e}")

        # Fallback: synthetic stubs
        print(f"GSM8K: generating {n} synthetic problems")
        for i in range(1, n + 1):
            problems.append({
                "id": f"gsm8k_{i}",
                "prompt": f"Natalia sold clips to {i*10} of her friends in April...",
                "expected": [str(i*10 + (i*10)//2)],
                "dataset": "gsm8k",
                "metadata": {"category": "synthetic", "is_stub": True}
            })
        return problems

    @staticmethod
    def load_math(n: int = 100) -> List[Dict]:
        """Loads MATH (Mathematics for Machine Learning) problems.

        Tries multiple dataset sources in order:
        1. 'hendrycks/competition_math' — correct mirror
        2. 'competition_math' — original (may not exist)
        3. LOCAL FILE: checks 'data/math_test.json' if available
        4. Stub fallback

        Args:
            n: Max problems to load. Default 100 (level 1-3).
        """
        problems = []

        sources = [
            "hendrycks/competition_math",
            "competition_math",
        ]

        if HAS_HF_DATASETS:
            for source in sources:
                try:
                    ds = load_dataset(source, split="test", streaming=True)
                    iterator = iter(ds)
                    loaded = 0
                    for i in range(n * 3):  # Load extra to filter by difficulty
                        try:
                            item = next(iterator)
                            difficulty = item.get("difficulty", 1)
                            if difficulty > 3:  # Skip level 4-5
                                continue
                            problems.append({
                                "id": f"math_{len(problems)+1}",
                                "prompt": item["problem"],
                                "expected": [item["solution"]],
                                "dataset": "math",
                                "metadata": {
                                    "difficulty": difficulty,
                                    "subject": item.get("subject", "unknown"),
                                }
                            })
                            loaded += 1
                            if loaded >= n:
                                break
                        except StopIteration:
                            break
                    if problems:
                        print(f"MATH: loaded {len(problems)} problems from {source}")
                        return problems
                except Exception as e:
                    print(f"MATH: {source} failed: {e}")
                    continue

        # Try local file
        local_path = Path("data/math_test.json")
        if local_path.exists():
            try:
                data = json.loads(local_path.read_text())
                for i, item in enumerate(data[:n]):
                    problems.append({
                        "id": f"math_{i+1}",
                        "prompt": item["problem"],
                        "expected": [item["solution"]],
                        "dataset": "math",
                        "metadata": {"difficulty": item.get("difficulty", 1)}
                    })
                print(f"MATH: loaded {len(problems)} from local file")
                return problems
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning("MATH local file load failed: %s", e)
                pass

        # Final fallback: synthetic
        print(f"MATH: generating {n} synthetic problems (all loaders failed)")
        for i in range(1, n + 1):
            problems.append({
                "id": f"math_{i}",
                "prompt": f"Solve for x: {i}x + {i*2} = {i*3 + i*2}",
                "expected": [str((i*3 + i*2 - i*2) / i) if i != 0 else "0"],
                "dataset": "math",
                "metadata": {"is_stub": True}
            })
        return problems

    @staticmethod
    def load_humaneval(n: int = 10) -> List[Dict]:
        """Loads HumanEval (Python coding problems)."""
        problems = []
        if HAS_HF_DATASETS:
            try:
                ds = load_dataset("openai_humaneval", split="test", streaming=True)
                iterator = iter(ds)
                for i in range(n):
                    try:
                        item = next(iterator)
                        problems.append({
                            "id": f"he_{i+1}",
                            "prompt": item["prompt"],
                            "expected": [item["canonical_solution"]],
                            "dataset": "humaneval",
                            "test": item.get("test", ""),
                            "entry_point": item.get("entry_point", ""),
                            "task_id": item.get("task_id", f"he_{i+1}"),
                        })
                    except StopIteration:
                        break
                return problems
            except Exception as e:
                logger.warning(f"Failed to load HumanEval from HuggingFace: {e}. Falling back to stubs.")

        # Fallback Stub
        for i in range(1, n + 1):
            problems.append({
                "id": f"he_{i}",
                "prompt": f"def add_numbers(a, b):\n    \"\"\" Add two numbers {i} times. \"\"\"",
                "expected": ["return (a + b)"],
                "dataset": "humaneval",
                "test": "def check(f): assert f(1, 2) == 3; assert f(-1, 1) == 0",
                "entry_point": "add_numbers",
                "task_id": f"he_{i}",
            })
        return problems

    @staticmethod
    def load_scibench(n: int = 10) -> List[Dict]:
        """Simulates loading SciBench (Scientific reasoning)."""
        # SciBench is not standard on HF or requires specific access/config usually.
        # Keeping as stub for reliability unless we find a specific HF path.
        # Attempting 'mit-han-lab/scibench' sometimes works but can be flaky.
        # We will stick to stub for now to avoid errors, or try a generic science dataset.
        problems = []
        for i in range(1, n + 1):
            problems.append({
                "id": f"scibench_{i}",
                "prompt": f"Calculate the kinetic energy of a {i}kg object moving at 10 m/s.",
                "expected": [str(0.5 * i * 100)],
                "dataset": "scibench"
            })
        return problems

def get_all_datasets(n_per_set: int = 5) -> List[Dict]:
    return (
        DatasetLoader.load_gsm8k(n_per_set) +
        DatasetLoader.load_math(n_per_set) +
        DatasetLoader.load_humaneval(n_per_set) +
        DatasetLoader.load_scibench(n_per_set)
    )
