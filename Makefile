# Variables
PY_SRC := holomotion/  # Your Python code directory
RUFF := ruff  # Assumes Ruff is installed locally
PYTHON ?= python
PYTEST := $(PYTHON) -m pytest -v
TRAIN_TESTS := tests/
UNITREE_TESTS := deployment/unitree_g1_ros2_29dof/tests/
UNITREE_SRC := deployment/unitree_g1_ros2_29dof/src
TELEOP_TESTS := deployment/holomotion_teleop/tests/
COV := --cov=holomotion/src --cov-report=term-missing

# Directory to lint/format - can be overridden with DIR=path
DIR ?= holomotion/src

.PHONY: lint format check lint-dir format-dir test

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
	OMNI_KIT_ACCEPT_EULA=YES $(PYTEST) $(TRAIN_TESTS)
	PYTHONPATH=$(UNITREE_SRC):$${PYTHONPATH} \
		OMNI_KIT_ACCEPT_EULA=YES $(PYTEST) $(UNITREE_TESTS)
	OMNI_KIT_ACCEPT_EULA=YES $(PYTEST) $(TELEOP_TESTS)
