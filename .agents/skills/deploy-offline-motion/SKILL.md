---
name: deploy-offline-motion
description: Deploy and troubleshoot a HoloMotion policy for offline real-robot motion tracking. Use for deployment preflight, no-action validation, model or motion substitution, controller startup, and offline robot-runtime diagnosis.
---

# Deploy Offline Motion

Treat the current repository documentation, deployment profile, launcher, and
runtime CLI as the source of truth. Discover their current locations and
interfaces before constructing commands; do not reuse paths or flags from a
remembered release.

## Safety boundary

Separate no-action validation from any command that can publish control. Never
start an actuating runtime or change robot mode without explicit user
confirmation immediately before the action. State the exact resolved command,
target robot, artifact, motion source, fixture state, and stop method.

## Workflow

1. Resolve the supported robot contract, DOF ordering, runtime environment,
   deployment profile, network interface, controller states, and stop path.
2. Resolve the current no-action preflight from the CLI help, launcher, and
   documentation. Verify artifact readability, schemas, providers, dimensions,
   and configuration without starting the controller.
3. Continue only when the preflight explicitly succeeds and evidence confirms
   that no action publisher or robot process was started.
4. Hold model, motion, profile, and runtime constant for the first active test.
   Prefer repository-provided baseline artifacts before substitutions.
5. Obtain confirmation immediately before the resolved actuating command.
6. Observe every controller transition and stop on unexpected state or motion.

## Custom motion

Resolve the current motion schema from the checked-in specification, loader,
and validator. Check required fields, shapes, ordering, units, rotations,
timestamps, finite values, and semantic plausibility. Validate the same file
with the deployed loader and test it in visualization or simulation before
real-robot use.

Run the bundled motion successfully before substituting a custom clip or custom model. Change one variable at a time.

## Diagnose by stage

Localize failures in this order:

1. package or image and target architecture;
2. accelerator and container runtime;
3. resolved profile, mounts, and network;
4. model loading and execution provider;
5. motion schema and preprocessing;
6. controller readiness and state transition;
7. policy timing and output validity;
8. physical behavior.

Do not bypass a failed no-action gate. Report the last successful stage, first
failing boundary, exact resolved artifacts and interfaces, relevant evidence,
and whether any robot action was sent.
