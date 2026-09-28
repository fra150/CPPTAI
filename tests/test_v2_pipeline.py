import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.pipeline_v2 import run as run_v2
from cpptai.types import RunConfig


class TestV2Pipeline(unittest.TestCase):
    def test_v2_writes_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(
                seed=0,
                external_enabled=False,
                cache_mode="offline",
                output_dir=td,
                benchmark_name="test",
                model_name="DeepSeek-V3.2-Exp",
                max_iterations=10,
            )
            result, artifact = run_v2("A simple test problem.", cfg)
            self.assertTrue(artifact.run_id)
            path = os.path.join(td, f"run_{artifact.run_id}.json")
            self.assertTrue(os.path.exists(path))
            self.assertIn("final_answer", result)

    def test_v2_deterministic_offline(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(
                seed=0,
                external_enabled=False,
                cache_mode="offline",
                output_dir=td,
                benchmark_name="test",
                model_name="DeepSeek-V3.2-Exp",
                max_iterations=10,
            )
            r1, a1 = run_v2("Determinism test.", cfg)
            r2, a2 = run_v2("Determinism test.", cfg)
            self.assertEqual(r1.get("final_answer"), r2.get("final_answer"))


if __name__ == "__main__":
    unittest.main()
