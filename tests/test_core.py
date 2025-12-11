import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.core import EntropicSegregator, DescentVector, CPPTAITraslocatore, ConvergenceProtocol
from cpptai import ProblemBlock, DifficultyLevel


class TestCPPTAICore(unittest.TestCase):
    def test_spectral_scan_and_segregation(self):
        seg = EntropicSegregator()
        problem = "Sentence one. Sentence two is longer. Short."
        blocks = seg.segregate(problem)
        self.assertGreaterEqual(len(blocks), 3)
        # Ensure blocks are ProblemBlock instances and ordered by heuristic
        self.assertIsInstance(blocks[0], ProblemBlock)
        self.assertTrue(0.0 <= blocks[0].complexity_score <= 1.0)

    def test_descent_vector_progress(self):
        dv = DescentVector()
        initial = {"problem": "X", "block_solutions": [], "building_height": 3}
        result = dv.cognitive_descent(3, initial)
        ans = result.get("final_answer", "")
        self.assertIn("confidence", ans)
        # Check coherence/completeness/confidence increased beyond initial 0.2
        last_state = result["descent_log"][-1]["state"]
        for key in ("coherence", "completeness", "confidence"):
            self.assertGreater(last_state[key], 0.2)

    def test_convergence_protocol_order(self):
        cp = ConvergenceProtocol()
        # Monkeypatch deepseek_chat in core via method override by subclassing
        def fake_query_divergent(ctx):
            return {"source": "deepseek", "content": "Test content", "confidence": 0.9}
        cp._query_divergent_twin = lambda ctx: fake_query_divergent(ctx)
        res = cp.convene_meeting({"problem": "Test"})
        self.assertIn("external_synthesis", res)
        self.assertIn("DeepSeek", res["external_synthesis"])  # label added in synthesis

    def test_orchestrator_runs(self):
        orchestrator = CPPTAITraslocatore()
        result = orchestrator.solve("A simple test problem.")
        self.assertIn("final_answer", result)
        self.assertIn("final_arranged", result)


if __name__ == "__main__":
    unittest.main()
