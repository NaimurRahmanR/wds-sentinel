.PHONY: setup lint test reproduce integrated

setup:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts

test:
	pytest -q

integrated:
	python scripts/run_hazard_integration_scenarios.py

reproduce:
	python scripts/reproduce_core_results.py
