# Git hooks

Pre-commit runs [ruff](https://docs.astral.sh/ruff/) format on staged Python files (using `train.env` for the correct environment).

**Install ruff** in the holomotion train environment if it is absent:

```bash
conda activate holomotion_train
pip install ruff
```

Ruff is also listed in `environments/requirements_base.txt` if you install deps from there.

**Enable hooks** (run once from repo root):

```bash
git config core.hooksPath .githooks
```

Ensure the hook is executable: `chmod +x .githooks/pre-commit`

**Requirement:** Run `git commit` from a shell where conda is available so `train.env` can set `Train_CONDA_PREFIX`.

**Skip hook for one commit:** `git commit --no-verify`
