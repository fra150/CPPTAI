import sys
import os
import tempfile
import unittest
import json
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.pipeline_v2 import run as run_v2
from cpptai.types import RunConfig


class TestExplorerAnalyzerIntegration(unittest.TestCase):

    def test_full_pipeline_with_explorer_analyzer(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(
                seed=0,
                explorer_enabled=True,
                analyzer_enabled=True,
                external_enabled=False,
                cache_mode="offline",
                output_dir=td,
                benchmark_name="test",
                model_name="DeepSeek-V3.2-Exp",
                max_iterations=10,
            )
            result, artifact = run_v2("Integration test problem.", cfg)
            self.assertIn("final_answer", result)
            self.assertIsNotNone(artifact.explorer_trajectories_count)
            self.assertIsNotNone(artifact.analyzer_confidence)
            phase_names = [p.name for p in artifact.phase_outputs]
            self.assertIn("phase0_explorer", phase_names)
            self.assertIn("phase0_analyzer", phase_names)

    def test_pipeline_without_explorer_backward_compat(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(
                seed=0,
                explorer_enabled=False,
                analyzer_enabled=False,
                external_enabled=False,
                cache_mode="offline",
                output_dir=td,
                benchmark_name="test",
                model_name="DeepSeek-V3.2-Exp",
                max_iterations=10,
            )
            result, artifact = run_v2("Backward compat problem.", cfg)
            phase_names = [p.name for p in artifact.phase_outputs]
            self.assertNotIn("phase0_explorer", phase_names)
            self.assertNotIn("phase0_analyzer", phase_names)

    def test_deterministic_offline_with_explorer(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(
                seed=42,
                explorer_enabled=True,
                analyzer_enabled=True,
                external_enabled=False,
                cache_mode="offline",
                output_dir=td,
                benchmark_name="test",
                model_name="DeepSeek-V3.2-Exp",
                max_iterations=10,
            )
            r1, a1 = run_v2("Determinism explorer test.", cfg)
            r2, a2 = run_v2("Determinism explorer test.", cfg)
            self.assertEqual(r1.get("final_answer"), r2.get("final_answer"))

    def test_artifact_json_serializable(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(
                seed=0,
                explorer_enabled=True,
                analyzer_enabled=True,
                external_enabled=False,
                cache_mode="offline",
                output_dir=td,
                benchmark_name="test",
                model_name="DeepSeek-V3.2-Exp",
                max_iterations=10,
            )
            result, artifact = run_v2("JSON serialization test.", cfg)
            payload = asdict(artifact)
            payload["phase_outputs"] = [asdict(p) for p in artifact.phase_outputs]
            json_str = json.dumps(payload, ensure_ascii=False, indent=2)
            self.assertTrue(json_str)
            decoded = json.loads(json_str)
            self.assertEqual(decoded["run_id"], artifact.run_id)


if __name__ == "__main__":
    unittest.main()
