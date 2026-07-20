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

Isaac Lab source is installed under the environment selected by `train.env`.
Resolve the Python version instead of hardcoding it:

```bash
cd <holomotion-root>
source train.env
pyver=$("$Train_CONDA_PREFIX/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
source_root="$Train_CONDA_PREFIX/lib/$pyver/site-packages/isaaclab/source"
```

## Resolving the path

If the computed path does not exist, search only under
`"$Train_CONDA_PREFIX/lib/"` for `site-packages/isaaclab/source` and report the
resolved path. Do not assume a global Conda base or Python minor version.

## How to look up source

1. **Do not import isaaclab.** Resolve the installation with filesystem paths.
2. **Use filesystem tools only:** prefer `rg`, file listing, and targeted reads.
3. **Map module to path:** e.g. `isaaclab.envs` → `source/isaaclab/isaaclab/envs/`, `isaaclab_rl` → `source/isaaclab_rl/`.
4. **Search by symbol or topic:** Use Grep over the source directory for class/function names or keywords.

## Example

To find how a VecEnv is created in Isaac Lab RL:

1. Resolve `source_root` with the environment Python version.
2. Grep for "VecEnv" or "vec_env" under that root.
3. Read the relevant files with the Read tool.

## Rules

- Never `import isaaclab` (or subpackages) to get source info; use file-based lookup only.
- Always use the `source/` subtree; the rest of site-packages may be generated or installed wrappers.
- Cite the source file and relevant symbol in the result.
