import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.presentation import arrange_solution_simple


class TestPresentation(unittest.TestCase):
    def test_arrange_technical(self):
        text = "Alpha. Beta. Gamma."
        arranged = arrange_solution_simple(text, context="technical")
        self.assertIn("Solution Report", arranged)
        self.assertIn("Summary", arranged)
        self.assertIn("Key Points", arranged)
        self.assertIn("Conclusion", arranged)

    def test_arrange_executive(self):
        text = "Implement storage. Reduce costs. Evaluate SMRs."
        arranged = arrange_solution_simple(text, context="executive")
        self.assertIn("Executive Summary", arranged)
        self.assertIn("Key Points", arranged)
        self.assertIn("Recommended Actions", arranged)


if __name__ == "__main__":
    unittest.main()
