import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.presentation import arrange_solution_simple


class TestPresentation(unittest.TestCase):
    def test_arrange_technical(self):
        text = "Alpha. Beta. Gamma."
        arranged = arrange_solution_simple(text, context="technical")
        self.assertIn("## Analysis", arranged)
        self.assertIn("## Solution", arranged)
        self.assertIn("## Details", arranged)

    def test_arrange_executive(self):
        text = "Implement storage. Reduce costs. Evaluate SMRs."
        arranged = arrange_solution_simple(text, context="executive")
        self.assertIn("KEY POINTS", arranged)
        self.assertIn("ACTIONS", arranged)


if __name__ == "__main__":
    unittest.main()
