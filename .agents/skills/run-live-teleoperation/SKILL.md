---
name: run-live-teleoperation
description: Set up, run, validate, or troubleshoot a supported HoloMotion live-teleoperation pipeline. Use for source readiness, reference processing, streaming, timing, visualization, policy integration, controller transitions, and real-robot teleoperation problems.
---

# Run Live Teleoperation

Resolve the currently supported pipeline from repository documentation,
profiles, launchers, and runtime code. Express the selected pipeline as:

```text
live source -> canonical reference -> optional transform -> observation -> policy -> controller
```

Do not infer where a transform runs or substitute a legacy path from memory.
Verify each boundary and keep visualization outside the control-critical path.

## Prerequisites

Before live teleoperation:

1. Complete the current no-action deployment preflight.
2. Validate the baseline offline policy on the same target and runtime.
3. Confirm the selected source, calibration, network, timestamps, and receiving service.
4. Confirm the resolved profile selects the intended source and processing path.
5. Prepare the robot in a safe fixture and keep the operator ready to stop it.

Ask for explicit user confirmation immediately before starting any command that
can send actions or changing robot mode.

## Bring up the reference path

1. Resolve and present the exact source and robot-side commands.
2. Start the robot-side command only after confirmation.
3. Connect the selected live source to the resolved service.
4. Confirm required streams, timestamps, reference validity, and queue readiness.
5. Enter motion tracking only after the runtime's readiness gates pass.
6. Use the optional workstation viewer only as telemetry; never put it in the control-critical path.

Do not bypass stale-data, queue-readiness, or timing protections to make the robot enter motion mode.

## Diagnose by stage

Find the first broken boundary:

1. source devices and calibration;
2. source service startup;
3. network reachability and source timestamps;
4. source payload and canonical-reference validity;
5. optional transform output and timing;
6. reference queue freshness;
7. observation construction and policy inference;
8. controller mode transition;
9. real-robot response.

Use timing and debug output already provided by the runtime. Keep visualization, recording, and logging non-blocking relative to the control path.

For real-robot behavior, treat the operator's confirmation as the result.
Report the resolved pipeline, first failing boundary, evidence, timing, safety
state, and any deviation from the currently supported path.
