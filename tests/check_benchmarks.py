import sys
import os
sys.path.insert(0, os.path.abspath("src"))
from cpptai.benchmarks import build_problems, rubric_accuracy

def test_benchmark_setup():
    print("Building problems...")
    probs = build_problems(10)
    print(f"Generated {len(probs)} problems.")
    
    # Verify we have mixed types
    datasets = set(p.get("dataset") for p in probs)
    print(f"Datasets found: {datasets}")
    assert "energy_synthetic" in datasets
    assert "gsm8k" in datasets or "math" in datasets
    
    # Test rubric
    print("Testing rubric...")
    # Exact match
    score = rubric_accuracy("The answer is 42.", ["42"])
    print(f"Rubric '42' in 'The answer is 42.': {score}")
    assert score == 1.0
    
    # Numeric tolerance
    score = rubric_accuracy("Value is 3.14159", ["3.14"])
    print(f"Rubric '3.14' in 'Value is 3.14159': {score}")
    assert score == 1.0
    
    print("Benchmark setup verified.")

if __name__ == "__main__":
    test_benchmark_setup()
