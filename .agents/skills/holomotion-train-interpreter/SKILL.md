---
name: holomotion-train-interpreter
description: Run HoloMotion Python scripts or inspect packages with the project training environment. Use for training, evaluation, motion conversion, project Python commands, dependency checks, or any task that must use the environment selected by train.env.
---

# HoloMotion training interpreter

## Workflow

1. Resolve the public HoloMotion Git root rather than assuming the current directory.
2. Source `<holomotion-root>/train.env` so `Train_CONDA_PREFIX` is set.
3. Use binaries under `"$Train_CONDA_PREFIX/bin/"`.

```bash
cd <holomotion-root>
source train.env
"$Train_CONDA_PREFIX/bin/python" path/to/script.py [args...]
```

```bash
"$Train_CONDA_PREFIX/bin/python" -m pip list
"$Train_CONDA_PREFIX/bin/python" -m pip show <package>
```

```bash
"$Train_CONDA_PREFIX/bin/python" -c "import sys; print(sys.executable)"
```

## Rules

- Do not rely on `python`, `pip`, or `conda activate` from the caller's shell.
- Verify `Train_CONDA_PREFIX` is non-empty and the requested binary exists.
- Preserve the caller's arguments and working-directory requirements.
- Do not import simulator packages merely to find their source; use the `isaaclab-source-lookup` skill.
