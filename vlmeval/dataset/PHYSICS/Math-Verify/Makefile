.PHONY: quality style test

check_dirs := src examples

check:
	ruff check $(check_dirs)  # linter
	ruff format --check $(check_dirs) # formatter

format:
	ruff check --fix $(check_dirs) # linter
	black $(check_dirs) # formatter

test:
	python -m pytest -sv ./tests/