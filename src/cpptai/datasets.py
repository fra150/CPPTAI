"""Dataset loaders for broader benchmarking (GSM8K, MATH, HumanEval, SciBench).
Currently provides stubs and synthetic generators, as actual datasets require external files.
"""

from typing import List, Dict, Optional
import logging
import random

try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

logger = logging.getLogger(__name__)

class DatasetLoader:
    @staticmethod
    def load_gsm8k(n: int = 10) -> List[Dict]:
        """Loads GSM8K (Grade School Math 8K) problems."""
        problems = []
        if HAS_HF_DATASETS:
            try:
                # Load streaming to avoid full download
                ds = load_dataset("gsm8k", "main", split="test", streaming=True)
                iterator = iter(ds)
                for i in range(n):
                    try:
                        item = next(iterator)
                        problems.append({
                            "id": f"gsm8k_{i+1}",
                            "prompt": item["question"],
                            "expected": [item["answer"].split("####")[-1].strip()],
                            "dataset": "gsm8k"
                        })
                    except StopIteration:
                        break
                return problems
            except Exception as e:
                logger.warning(f"Failed to load GSM8K from HuggingFace: {e}. Falling back to stubs.")
        
        # Fallback Stub
        for i in range(1, n + 1):
            problems.append({
                "id": f"gsm8k_{i}",
                "prompt": f"Natalia sold clips to {i*10} of her friends in April, and then she sold half as many in May. How many clips did she sell altogether in April and May?",
                "expected": [str(i*10 + (i*10)//2)],
                "dataset": "gsm8k"
            })
        return problems

    @staticmethod
    def load_math(n: int = 10) -> List[Dict]:
        """Loads MATH (Mathematics for Machine Learning) problems."""
        problems = []
        if HAS_HF_DATASETS:
            try:
                ds = load_dataset("competition_math", split="test", streaming=True, trust_remote_code=True)
                iterator = iter(ds)
                for i in range(n):
                    try:
                        item = next(iterator)
                        problems.append({
                            "id": f"math_{i+1}",
                            "prompt": item["problem"],
                            "expected": [item["solution"]],
                            "dataset": "math"
                        })
                    except StopIteration:
                        break
                return problems
            except Exception as e:
                logger.warning(f"Failed to load MATH from HuggingFace: {e}. Falling back to stubs.")

        # Fallback Stub
        for i in range(1, n + 1):
            problems.append({
                "id": f"math_{i}",
                "prompt": f"Let f(x) = {i}x + 2. Find f(3).",
                "expected": [str(i*3 + 2)],
                "dataset": "math"
            })
        return problems

    @staticmethod
    def load_humaneval(n: int = 10) -> List[Dict]:
        """Loads HumanEval (Python coding problems)."""
        problems = []
        if HAS_HF_DATASETS:
            try:
                ds = load_dataset("openai_humaneval", split="test", streaming=True, trust_remote_code=True)
                iterator = iter(ds)
                for i in range(n):
                    try:
                        item = next(iterator)
                        problems.append({
                            "id": f"he_{i+1}",
                            "prompt": item["prompt"],
                            "expected": [item["canonical_solution"]],
                            "dataset": "humaneval"
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
                "dataset": "humaneval"
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
