# HoloMotion Agent Instructions

## Use project skills automatically

Inspect the skill descriptions under `.agents/skills/` and use every skill that clearly matches the user's task. Do not wait for the user to name a skill.

- Start with the task-level workflow skill, such as motion preparation, training, deployment, teleoperation, or diagnosis.
- Use supporting utility skills for the training interpreter, Isaac Lab source lookup, and Ruff formatting when needed.
- Combine skills when a task crosses workflow stages.
- Read current source, configuration, and linked documentation before relying on remembered commands.
- If no skill matches, continue with normal repository analysis.

## Confirmation and validation

- Never commit, push, tag, publish, or release without explicit user confirmation.
- Never run a command that can send actions to a real robot without explicit user confirmation immediately before that action.
- Distinguish static checks, simulation results, runtime checks, and user-confirmed real-robot results.
- Preserve unrelated user changes.
