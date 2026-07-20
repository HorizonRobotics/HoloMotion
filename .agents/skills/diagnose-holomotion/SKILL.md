---
name: diagnose-holomotion
description: Diagnose HoloMotion errors across environment setup, motion conversion, retargeting, training, evaluation, export, simulation, offline deployment, and live teleoperation. Use when a command fails, hangs, produces invalid motion or policy output, differs between stages, or has an unknown HoloMotion root cause.
---

# Diagnose HoloMotion

The goal is either to solve a known failure or to localize an unknown failure to one stage with reproducible evidence.

## Establish the failing boundary

Collect:

- exact command and working directory;
- relevant environment and config names without secrets;
- complete first error, not only the final wrapper error;
- input and output paths and schemas;
- last known working revision or artifact;
- expected versus observed behavior;
- whether the failure is deterministic;
- whether robot actions were involved.

Classify the failure:

1. environment and dependencies;
2. raw motion parsing;
3. HoloSMPL canonical/formal conversion;
4. HoloRetarget;
5. robot HDF5 loading;
6. training;
7. checkpoint evaluation;
8. PyTorch-to-ONNX export;
9. MuJoCo sim2sim;
10. Docker and robot runtime;
11. offline reference;
12. teleoperation reference;
13. policy inference or physical response.

Do not debug later stages until their inputs are validated.

## Reduce the problem

1. Read the actual entry script and selected config.
2. Reproduce with the smallest representative input and resource count.
3. Validate shapes, dtypes, units, coordinate frames, joint order, frame rate, and metadata at the failing boundary.
4. Compare a failing sample with a known-good public sample.
5. Find the first divergent artifact when two backends or stages disagree.
6. Search source by the exact exception, symbol, or log message.
7. Change one variable at a time and preserve the command and evidence.

Use the `holomotion-train-interpreter` skill for training-environment Python. Use `isaaclab-source-lookup` for Isaac Lab implementation questions.

## Safety and cost

- Prefer read-only inspection and no-action checks.
- Do not submit remote jobs or start costly training without explicit confirmation.
- Never operate the robot or bypass deployment safety checks without explicit confirmation.
- Do not ask users to publish private data, credentials, checkpoints, or machine-specific paths.

## Report

Report:

- symptom;
- last successful stage;
- first failing stage;
- root cause or bounded hypotheses;
- evidence for each conclusion;
- minimal fix or next discriminating check;
- validation performed;
- remaining runtime or real-robot validation.
