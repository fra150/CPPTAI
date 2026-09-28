"""Real HumanEval execution verifier using subprocess.
Replaces the proxy rubric-based verifier with actual test execution.
"""
from __future__ import annotations
import ast
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from typing import Dict, List, Optional, Tuple

HUMANEVAL_TIMEOUT = 10  # seconds per test

def verify_humaneval(
    generated_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = HUMANEVAL_TIMEOUT,
) -> Dict:
    """Execute the generated code against HumanEval test cases.

    Args:
        generated_code: The LLM-generated solution.
        test_code: The HumanEval test assertions.
        entry_point: Function name to test.
        timeout: Max seconds per execution.

    Returns:
        Dict with:
        - passed: bool
        - execution_error: Optional error message
        - stdout: captured stdout
        - timing: execution time in seconds
    """
    import time

    # Build complete test script
    test_script = textwrap.dedent(f"""
import sys
import traceback

# Generated solution
{generated_code}

# Test cases
{test_code}

# Run test
if __name__ == "__main__":
    try:
        check({entry_point})
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {{e}}")
    except Exception as e:
        print(f"ERROR: {{e}}")
        traceback.print_exc()
""")

    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.py")
            with open(script_path, "w") as f:
                f.write(test_script)

            result = subprocess.run(
                [sys.executable, "-I", "-u", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""},
            )

            dt = time.perf_counter() - t0
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            passed = "PASSED" in stdout
            error = stderr if stderr else None

            if "FAILED" in stdout:
                error = stdout

            return {
                "passed": passed,
                "execution_error": error,
                "stdout": stdout[:500],
                "timing": round(dt, 3),
            }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "execution_error": f"Timeout after {timeout}s",
            "stdout": "",
            "timing": timeout,
        }
    except Exception as e:
        return {
            "passed": False,
            "execution_error": str(e),
            "stdout": "",
            "timing": round(time.perf_counter() - t0, 3),
        }


def is_valid_python(code: str) -> Tuple[bool, Optional[str]]:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
