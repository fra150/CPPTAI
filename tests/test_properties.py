"""Property-based tests for CPPTAI core components.
Uses random fuzzing to ensure robustness against diverse inputs.
"""

import unittest
import random
import string
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cpptai.core import EntropicSegregator, DescentVector, CPPTAITraslocatore, ConvergenceProtocol

def random_string(length: int = 50) -> str:
    """Generate a random string of fixed length."""
    letters = string.ascii_letters + string.digits + " .!?"
    return ''.join(random.choice(letters) for _ in range(length))

class TestProperties(unittest.TestCase):
    def test_entropic_segregator_robustness(self):
        """Fuzz EntropicSegregator with random strings."""
        seg = EntropicSegregator()
        for _ in range(20):
            problem = random_string(random.randint(10, 500))
            try:
                blocks = seg.segregate(problem)
                self.assertIsInstance(blocks, list)
                for b in blocks:
                    self.assertTrue(0.0 <= b.complexity_score <= 1.0)
            except Exception as e:
                self.fail(f"EntropicSegregator crashed on input: {problem[:20]}... Error: {e}")

    def test_descent_vector_stability(self):
        """Ensure descent doesn't crash or diverge wildly on random numeric states."""
        dv = DescentVector()
        for _ in range(20):
            # Random initial state
            initial_context = {
                "problem": "Test",
                "coherence": random.random(),
                "completeness": random.random(),
                "confidence": random.random()
            }
            try:
                # Use a small height for speed
                result = dv.cognitive_descent(3, initial_context)
                final_state = result["descent_log"][-1]["state"]
                
                # Invariants: scores should remain in [0, 1] (or close to it if logic allows overflow, 
                # but core.py clamps them, so we verify that clamp works)
                for k in ("coherence", "completeness", "confidence"):
                    val = final_state.get(k, 0.0)
                    self.assertTrue(0.0 <= val <= 1.0, f"{k} out of bounds: {val}")
            except Exception as e:
                self.fail(f"DescentVector crashed on random state. Error: {e}")

    def test_end_to_end_fuzz(self):
        """Run full orchestrator on random garbage text."""
        orchestrator = CPPTAITraslocatore(enable_phase_iv=False) # Skip external to speed up
        for _ in range(5):
            problem = random_string(100)
            try:
                res = orchestrator.solve(problem)
                self.assertIn("final_answer", res)
                self.assertIn("responsible_ai_audit", res)
            except Exception as e:
                self.fail(f"Orchestrator crashed on input: {problem[:20]}... Error: {e}")

if __name__ == "__main__":
    unittest.main()
