.PHONY: install test lint format typecheck check clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=prism --cov-report=term-missing --cov-fail-under=85

lint:
	ruff check prism tests

format:
	ruff format prism tests

format-check:
	ruff format --check prism tests

typecheck:
	mypy prism

check: lint format-check typecheck test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
