---
name: isaaclab-source-lookup
description: Look up Isaac Lab source code by reading files under the installed isaaclab source tree. Use when Isaac Lab APIs, implementations, or source-level information are needed. Do not import isaaclab; use file reads and search on the source path instead.
---

# Isaac Lab source lookup

## When to use

Invoke this skill whenever:
- Isaac Lab source code, APIs, or implementation details are needed
- You need to understand how isaaclab/isaaclab_rl/isaaclab_tasks (etc.) implement something
- You are answering questions about Isaac Lab types, envs, or configs

## Source path

Isaac Lab source is installed under the holomotion_train conda env:

```
${Train_CONDA_PREFIX}/lib/python3.11/site-packages/isaaclab/source/
```

Subpackages under `source/` include:
- `isaaclab/` – core
- `isaaclab_assets/`
- `isaaclab_mimic/`
- `isaaclab_rl/`
- `isaaclab_tasks/`

## Resolving the path

`Train_CONDA_PREFIX` is set by sourcing the project env file:

- **From repo root:** `source holomotion/train.env`
- **From holomotion dir:** `source train.env`

Then the source root is: `"${Train_CONDA_PREFIX}/lib/python3.11/site-packages/isaaclab/source/"`.

When you cannot run shell (e.g. in Cursor), use the typical env path: `$CONDA_BASE/envs/holomotion_train` with `CONDA_BASE` from `conda info --base`, or assume the path is already correct if the user has sourced `train.env`.

## How to look up source

1. **Do not import isaaclab.** Importing isaaclab requires the simulator app to be launched first and will fail in normal editor/script contexts.
2. **Use filesystem tools only:** Read, Glob, and Grep on the source path above.
3. **Map module to path:** e.g. `isaaclab.envs` → `source/isaaclab/isaaclab/envs/`, `isaaclab_rl` → `source/isaaclab_rl/`.
4. **Search by symbol or topic:** Use Grep over the source directory for class/function names or keywords.

## Example

To find how a VecEnv is created in Isaac Lab RL:

1. Set source root: `"${Train_CONDA_PREFIX}/lib/python3.11/site-packages/isaaclab/source/"`.
2. Grep for "VecEnv" or "vec_env" under that root.
3. Read the relevant files with the Read tool.

## Rules

- Never `import isaaclab` (or subpackages) to get source info; use file-based lookup only.
- Always use the `source/` subtree; the rest of the site-packages tree may be generated or non-source.
- Prefer Read/Glob/Grep on the resolved path; run shell only when you need to resolve `Train_CONDA_PREFIX` (e.g. `source holomotion/train.env && echo $Train_CONDA_PREFIX`).
