# Variables
PY_SRC := holomotion/  # Your Python code directory
RUFF := ruff  # Assumes Ruff is installed locally
PYTEST := pytest -v
TESTS := holomotion/tests/
COV := --cov=holomotion/src --cov-report=term-missing

# Directory to lint/format - can be overridden with DIR=path
DIR ?= holomotion/src

.PHONY: lint format check lint-dir format-dir

# Run Ruff linter on default directory
lint:
	@echo "Linting with Ruff..."
	@$(RUFF) check $(PY_SRC)

# Format code in default directory
format:
	@echo "Formatting with Ruff..."
	@$(RUFF) format $(PY_SRC)
	@$(RUFF) check --fix $(PY_SRC)  # Auto-fix lint errors

# Run Ruff linter on specific directory (with fallback)
lint-dir:
	@echo "Linting directory: $(DIR)"
	@$(RUFF) check $(DIR)

# Format code in specific directory (with fallback)
format-dir:
	@echo "Formatting directory: $(DIR)"
	@$(RUFF) format $(DIR)
	@$(RUFF) check --fix $(DIR)  # Auto-fix lint errors

# Strict check (for CI)
check:
	@$(RUFF) check $(PY_SRC) --exit-non-zero-on-fix

# Run all tests
test:
	$(PYTEST) $(TESTS)