---
name: ruff-format
description: Runs project-wise Python formatting with ruff after sourcing train.env for the correct Python path. Use when the user asks to format Python code with ruff, run ruff, or run project-wide formatting.
---

# Ruff format (project-wide)

## When to use

Apply this skill when the user wants to:
- Run ruff for Python formatting
- Format project Python code with ruff
- Run project-wise or project-wide formatting

## Workflow

1. **Source the environment** (from repo root). This sets `Train_CONDA_PREFIX` so ruff and Python come from the holomotion_train conda env.
2. **Run ruff** using that env’s binary: `"$Train_CONDA_PREFIX/bin/ruff"`.

**Format (write changes), from repo root:**
```bash
source train.env && "$Train_CONDA_PREFIX/bin/ruff" format --config pyproject.toml ./
```

**Format check only (no write), from repo root:**
```bash
source train.env && "$Train_CONDA_PREFIX/bin/ruff" format --check --config pyproject.toml ./
```

## Rules

- Always source `train.env` before running ruff.
- Use `$Train_CONDA_PREFIX/bin/ruff` so the correct env is used.
- Ruff config is in `pyproject.toml`; from repo root pass `--config pyproject.toml` and target `./`.
