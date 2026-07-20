---
name: run-live-teleoperation
description: Set up, run, validate, or troubleshoot HoloMotion live teleoperation with PICO/XRoboToolkit and on-robot HoloRetarget. Use for the supported real-time teleoperation path, VR readiness, reference streaming, timing, optional visualization, or real-robot teleoperation problems.
---

# Run Live Teleoperation

Follow the supported v1.4 path in `docs/realworld_deployment.md`:

```text
PICO / XRoboToolkit -> on-robot HoloRetarget -> observation -> policy -> robot
```

Do not substitute the legacy workstation-retarget path described in `deployment/holomotion_teleop/holomotion_teleop_setup.md` unless the user explicitly requests development or debugging of that legacy path.

## Prerequisites

Before live teleoperation:

1. Complete `holomotion check`.
2. Run the bundled offline motion successfully on the same robot and deployment image.
3. Confirm the supported PICO/XRoboToolkit setup, full-body calibration, network, and robot-side service.
4. Confirm the launch profile uses the supported local reference source and enables teleoperation.
5. Prepare the robot in a safe fixture and keep the operator ready to stop it.

`holomotion teleop` can send robot actions. Ask for explicit user confirmation immediately before running it or changing robot mode.

## Bring up the reference path

1. Start the robot-side teleoperation command only after confirmation.
2. Point the XRoboToolkit PICO app at the robot-side service.
3. Confirm the service status and required body streams.
4. Wait for the reference queue to become ready before entering motion tracking.
5. Use the optional workstation viewer only as telemetry; never put it in the control-critical path.

Do not bypass stale-data, queue-readiness, or timing protections to make the robot enter motion mode.

## Diagnose by stage

Find the first broken boundary:

1. PICO devices and calibration;
2. XRoboToolkit service startup;
3. network reachability and source timestamps;
4. body-pose input validity;
5. HoloRetarget output and timing;
6. reference queue freshness;
7. observation construction and policy inference;
8. controller mode transition;
9. real-robot response.

Use timing and debug output already provided by the runtime. Keep visualization, recording, and logging non-blocking relative to the control path.

For real-robot behavior, treat the user/operator's confirmation as the result. Report the first failing stage, evidence, timing observations, safety state, and any change from the supported deployment path.
