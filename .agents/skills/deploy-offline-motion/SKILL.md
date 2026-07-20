---
name: deploy-offline-motion
description: Deploy and troubleshoot the released HoloMotion model for offline motion tracking on a supported Unitree G1 robot. Use when users want to run the pretrained model, replay the bundled motion, mount a custom motion NPZ, perform the no-action deployment check, or diagnose offline real-robot startup.
---

# Deploy Offline Motion

Use the Docker workflow in `docs/realworld_deployment.md`. The supported public v1.4 path is the self-contained robot deployment image, not a host-side training environment.

## Safety boundary

`holomotion check` is no-action validation. `holomotion offline` can send robot actions.

Never start the container command that operates the robot, run `holomotion offline`, or change robot mode without explicit user confirmation. Before requesting confirmation, state the exact command, robot, motion file, operator stop method, and whether the robot is suspended or in a safe fixture.

## Workflow

1. Confirm the supported robot, DOF configuration, onboard runtime, Docker runtime, network interface, and remote controller are available.
2. Follow the image pull/load and container startup instructions in `docs/realworld_deployment.md`.
3. Run:

   ```bash
   holomotion check
   ```

4. Continue only when the check explicitly reports that it passed and no action was sent.
5. For the first active test, use the bundled model and bundled offline motion.
6. Confirm robot preparation and emergency-stop readiness.
7. Ask for explicit approval immediately before running:

   ```bash
   holomotion offline
   ```

8. Observe startup and controller transitions. Stop immediately on unexpected motion.

## Custom motion

Validate the custom NPZ before mounting it into the container. It must follow `docs/holomotion_motion_file_spec.md` and the deployment requirements in `docs/realworld_deployment.md`.

Check the six required `ref_*` arrays, consistent frame counts, finite values, expected 29-DOF and body layouts, quaternion order, units, frame rate metadata, and plausible joint motion. Test the custom clip in visualization or simulation before real-robot use.

Run the bundled motion successfully before substituting a custom clip or custom model. Change one variable at a time.

## Diagnose by stage

Localize failures in this order:

1. image and architecture;
2. Docker/NVIDIA runtime;
3. launch profile and network;
4. model and ONNX providers;
5. motion NPZ schema;
6. controller readiness and mode transition;
7. policy runtime;
8. physical behavior.

Do not bypass a failed `holomotion check`. Report the last successful stage, first failing stage, relevant logs, and whether any robot action was sent.
