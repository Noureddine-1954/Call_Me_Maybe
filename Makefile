install:
	uv sync

run:
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

debug:
	uv run python3 -m pdb -m src

clean:
	@echo "Cleaning cache files..."
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

lint:
	flake8 .
	mypy .

lint-strict:
	flake8 .
	mypy . --strict