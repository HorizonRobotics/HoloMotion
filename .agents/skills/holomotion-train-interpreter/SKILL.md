---
name: holomotion-train-interpreter
description: Select and use the task's HoloMotion Python environment. Use for training, evaluation, motion conversion, dependency checks, and project Python commands, including the standalone HoloBoxing workflow.
---

# HoloMotion training interpreter

## Select the task environment

For participant HoloBoxing data, MuJoCo training, and export, use the
[standalone boxing setup](../../../holoboxing/README.md#install-the-training-environment-optional).
Use that environment's explicit Python path, normally
`<repo>/.venv-holoboxing/bin/python`; do not clone the full Isaac training environment.
Optional human-motion retargeting uses the separate environment documented there,
because its Newton dependency requires a different Warp version.
Docker isolates evaluation dependencies; reference training still runs in the
host Python environment and enforces its physics dependency versions.

For HoloMotion tracker training and other workflows owned by `train.env`,
use the procedure below. Verify `sys.executable` before running either workflow.

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

- Do not rely on an unverified `python`, `pip`, or activation state from the caller's shell.
- For `train.env` workflows, verify `Train_CONDA_PREFIX` is non-empty and the requested binary exists.
- Preserve the caller's arguments and working-directory requirements.
- Do not import simulator packages merely to find their source; use the `isaaclab-source-lookup` skill.
