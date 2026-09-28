.PHONY: test benchmark benchmark-full benchmark-compare benchmark-explorer clean

test:
	python -m pytest tests/ -v --tb=short

benchmark:
	python scripts/run_full_suite.py --gsm8k 50 --math 20 --humaneval 10

benchmark-full:
	python scripts/run_full_suite.py --gsm8k 1319 --math 100 --humaneval 164

benchmark-compare:
	python scripts/run_full_suite.py --compare --gsm8k 100

benchmark-explorer:
	python scripts/run_full_suite.py --explorer --gsm8k 50

clean:
	rm -rf benchmarks/ __pycache__ .pytest_cache
