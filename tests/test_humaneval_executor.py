import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cpptai.humaneval_executor import verify_humaneval, is_valid_python

class TestHumanEvalExecutor(unittest.TestCase):
    def test_valid_code(self):
        valid, err = is_valid_python("def f(): return 1")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_code(self):
        valid, err = is_valid_python("def f(:")
        self.assertFalse(valid)
        self.assertIsNotNone(err)

    def test_correct_solution(self):
        code = "def add(a, b): return a + b"
        test = "def check(f): assert f(1, 2) == 3; assert f(-1, 1) == 0"
        result = verify_humaneval(code, test, "add")
        self.assertTrue(result["passed"])

    def test_incorrect_solution(self):
        code = "def add(a, b): return a - b"
        test = "def check(f): assert f(1, 2) == 3"
        result = verify_humaneval(code, test, "add")
        self.assertFalse(result["passed"])

if __name__ == "__main__":
    unittest.main()
