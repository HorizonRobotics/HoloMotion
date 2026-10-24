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

When MuJoCo observations disagree after stepping, inspect which integration stage produced each derived cache.
Compare independently refreshed poses and velocities from the same current joint state.
Keep policy refreshes separate from solver/contact data, and compare trajectories with fixed control inputs.
After reset, check buffer ownership, captured addresses, and actual container paths.

Generated observation bundles can select a different device-descriptor backend.
Reproduce training with the same bundle present, including the first real lower-controller inference.
Convert foreign framework device descriptors before Torch allocation and synchronization.
An import or CPU resource preflight does not exercise this boundary.

If GPU tests pass separately but fail together with `unknown stream`, inspect the previous test's active Warp stream.
Tests selecting a Torch stream must restore Warp's previous device and stream through cleanup scopes, including failed setup.
A leaked default stream can break a later CUDA graph capture.

After removing heavy initialization, test the real entrypoint without GPU access.
Keep a lightweight device check before loading native inference libraries.
Those libraries can terminate the process before Python catches an error.

Persist simulator SDK logs in owned output storage before startup, rather than assuming a container-local printed path survives.
Keep structured progress records separate from SDK severity records when scanning diagnostics.
A schema key mentioning overflow is not itself an overflow event.
If finalization fails after training and evaluation, retain their weights and traces and recover validation before retraining.
When a shared evaluator adds report fields, exercise report generation through every backend subclass.
Subclasses that defer simulator creation can bypass base initialization and fail only after the complete rollout.

For recurrent history overhead, separate KV reconstruction from unused action computation.
Omit a final FFN only when its output cannot affect subsequent KV and its discarded evaluation has no persistent or random side effects.
Compare padded and wrapped caches, next actions, auxiliary losses and gradients before using a faster burn-in path.
Keep prefill timings distinct from measured training throughput and task quality.

Before interpreting quality changes, repeat a frozen checkpoint with identical hardware, runtime, scene construction and inputs.
Compare per-frame actions, states, rewards and termination flags, then localize the first divergence.
Keep cross-hardware differences separate from same-device nondeterminism and from training variation.

For training throughput, inspect the joint collection, model, actor, critic, allocation, and synchronization timeline.
Verify effective runtime settings through their APIs instead of assuming environment variables were applied.
Device memory includes simulator allocations outside Torch's counters.
Preserve allocator failures, and compare successful candidates with identical logical updates in a profiler-free fixed-work interval.
Separate cold compilation, stable training, saving, and evaluation costs.

A zero-action startup check does not validate contact stability under sampled policy actions.
Audit resolved motor effort and velocity limits, then inspect the first actual rollout before interpreting learning losses.
Finite solver explosions require rejection and saved state evidence; a finite-value check alone is insufficient.
Restoring a simulator motor constraint can change engine semantics and must be disclosed in controlled comparisons.

For batched simulator geometry audits, separate reference-matmul precision from physical state precision.
Retain failing link transforms and compare residuals against environment origins and population size.
Large float32 world coordinates can exceed a local position tolerance after origin subtraction.
Check compact collision-isolated layouts before changing geometry tolerances or the learning method.

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
