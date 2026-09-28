"""Integration tests for benchmark automation."""
import sys
import os
import tempfile
import unittest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scripts.run_full_suite import BenchmarkOrchestrator


class TestBenchmarkAutomation(unittest.TestCase):
    def test_orchestrator_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            orch = BenchmarkOrchestrator(output_dir=td)
            self.assertTrue(os.path.exists(td))

    def test_gsm8k_small(self):
        with tempfile.TemporaryDirectory() as td:
            orch = BenchmarkOrchestrator(output_dir=td, workers=2)
            summary = orch.run_gsm8k(n=5)
            self.assertIn("accuracy", summary)
            self.assertGreaterEqual(len(orch.results), 1)

    def test_generate_reports_creates_files(self):
        with tempfile.TemporaryDirectory() as td:
            orch = BenchmarkOrchestrator(output_dir=td)
            orch.run_gsm8k(n=3)
            orch.generate_reports()
            files = os.listdir(td)
            csv_files = [f for f in files if f.endswith(".csv")]
            json_files = [f for f in files if f.endswith(".json")]
            self.assertTrue(len(csv_files) >= 1)
            self.assertTrue(len(json_files) >= 1)

    def test_group_by_method(self):
        with tempfile.TemporaryDirectory() as td:
            orch = BenchmarkOrchestrator(output_dir=td)
            orch.results = [
                {"method": "CPPTAI", "accuracy": 1.0, "dataset": "test"},
                {"method": "CPPTAI", "accuracy": 0.0, "dataset": "test"},
                {"method": "CPPTAI_EA", "accuracy": 1.0, "dataset": "test"},
            ]
            groups = orch._group_by_method()
            self.assertEqual(groups["CPPTAI"]["accuracy"], 0.5)
            self.assertEqual(groups["CPPTAI_EA"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
