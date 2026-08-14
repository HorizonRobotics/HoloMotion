---
name: isaaclab-source-lookup
description: Look up Isaac Lab APIs and implementations in the source tree associated with the project's active training environment. Use for source-level questions about Isaac Lab packages without importing simulator modules.
---

# Isaac Lab Source Lookup

1. Read the repository's current environment bootstrap and resolve the exact
   Python interpreter used by the selected training or evaluation workflow.
2. Derive that interpreter's site-packages location from Python itself; never
   assume a Conda root or Python minor version.
3. Locate the installed Isaac Lab `source` subtree with filesystem search. Do
   not import Isaac Lab or start simulator packages merely to discover source.
4. Map the requested module to its package under the source tree, then use
   `rg`, file listing, and targeted reads to find the symbol and its callers.
5. Verify behavior in implementation and call sites instead of relying only on
   type stubs, generated wrappers, or remembered APIs.
6. Cite the resolved source file and symbol, and report the environment from
   which it was resolved.

If multiple installations exist, use the one selected by the current workflow
and state the ambiguity rather than silently choosing a global environment.
