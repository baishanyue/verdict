PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test lint demo bench clean

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src tests benchmarks

demo:
	mkdir -p artifacts
	$(PYTHON) -m verdict run --cases examples/cases.jsonl --predictions examples/predictions.jsonl --suite demo --out artifacts/demo-run.json
	$(PYTHON) -m verdict report --run artifacts/demo-run.json --out artifacts/demo-report.md
	$(PYTHON) -m verdict report --run artifacts/demo-run.json --out artifacts/demo-report.html
	$(PYTHON) -m verdict compare --baseline artifacts/demo-run.json --candidate artifacts/demo-run.json

bench:
	$(PYTHON) benchmarks/bench.py

clean:
	rm -rf .pytest_cache build dist artifacts src/*.egg-info
	find src tests benchmarks -type d -name __pycache__ -prune -exec rm -rf {} +
