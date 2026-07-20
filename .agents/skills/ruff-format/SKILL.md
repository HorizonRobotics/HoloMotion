---
name: ruff-format
description: Format HoloMotion Python files with the Ruff binary selected by train.env. Use when the user asks to format Python code, run Ruff formatting, check formatting, format changed files, or explicitly format the whole project.
---

# Ruff Format

## Workflow

1. Resolve and enter the public HoloMotion Git root.
2. Source `train.env`.
3. Verify `"$Train_CONDA_PREFIX/bin/ruff"` exists.
4. Determine the requested scope.

Default to explicitly requested files. If the user asks to format current changes, derive the Python file list from staged and unstaged Git changes and exclude deleted files.

```bash
"$Train_CONDA_PREFIX/bin/ruff" format --config pyproject.toml <files...>
```

Use check-only mode when requested:

```bash
"$Train_CONDA_PREFIX/bin/ruff" format --check --config pyproject.toml <files...>
```

Run project-wide formatting only when the user explicitly requests it:

```bash
"$Train_CONDA_PREFIX/bin/ruff" format --config pyproject.toml .
```

After formatting, inspect the diff and report files changed. Never revert unrelated user changes.
