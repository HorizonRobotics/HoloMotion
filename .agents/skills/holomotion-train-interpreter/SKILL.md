---
name: holomotion-train-interpreter
description: Run Python scripts or inspect installed packages using the HoloMotion training conda env. Use when running project Python scripts, executing training-related code, or inspecting packages in the holomotion_train environment. Always source train.env first, then use Train_CONDA_PREFIX.
---

# HoloMotion training interpreter

## When to use

Invoke this skill when you need to:
- Run Python scripts with the project’s training environment (correct Python and deps)
- Execute training-related or holomotion Python code
- Inspect installed packages (e.g. `pip list`, `pip show`, site-packages) in the training env

## Workflow

1. **Source the environment** so `Train_CONDA_PREFIX` is set.
2. **Use that env’s Python/pip** via `$Train_CONDA_PREFIX/bin/python` and `$Train_CONDA_PREFIX/bin/pip`.

**Where to source from:**
- **From repo root:** `source holomotion/train.env`
- **From holomotion dir:** `source train.env`

**Run a Python script (from holomotion dir):**
```bash
source train.env && "$Train_CONDA_PREFIX/bin/python" path/to/script.py [args...]
```

**Run a Python script (from repo root):**
```bash
source holomotion/train.env && "$Train_CONDA_PREFIX/bin/python" holomotion/path/to/script.py [args...]
```

**Inspect installed packages:**
```bash
source train.env && "$Train_CONDA_PREFIX/bin/pip" list
source train.env && "$Train_CONDA_PREFIX/bin/pip" show <package>
```

**One-off Python (e.g. check version or import):**
```bash
source train.env && "$Train_CONDA_PREFIX/bin/python" -c "import sys; print(sys.executable)"
```

## Rules

- Always source `train.env` (or `holomotion/train.env` from repo root) before using `Train_CONDA_PREFIX`.
- Use `"$Train_CONDA_PREFIX/bin/python"` and `"$Train_CONDA_PREFIX/bin/pip"` so the correct env is used; avoid relying on whatever `python` is on PATH.
- For package inspection, use the same env (e.g. `$Train_CONDA_PREFIX/bin/pip list` or reading under `$Train_CONDA_PREFIX/lib/python3.11/site-packages/`).
