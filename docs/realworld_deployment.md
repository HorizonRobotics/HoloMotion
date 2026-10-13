# HoloMotion Real-World Deployment

This guide describes the Docker workflow for deploying HoloMotion v2.0 on a
Unitree G1 29-DOF robot with onboard NVIDIA Jetson Orin.

The `holomotion check` and `holomotion teleop` commands in this guide are
entry points supplied by the deployment image. Installing this source checkout
with `pip install -e .` does not install that CLI. Follow this guide with the
matching delivered image; for workstation simulation, use the
[evaluation scripts](evaluate_motion_tracking.md).

The deployment image is self-contained: it includes the robot-side code,
velocity policy, v2.0 motion-tracking policy, robot assets, and the matching
TensorRT engine cache. Users do not need to clone the repository or download
SMPL models for robot deployment. This image does not enable recording.

The existing two-policy controller and safety state machine are retained:
velocity tracking is used before entering motion tracking. The motion policy
currently exposes 1 or 5 reference control points and always outputs 29 joint
actions. A single policy replacing both controller modes is not part of this
deployment.

## Requirements

- Unitree G1 29-DOF robot with Jetson Orin.
- JetPack 5.1 compatible NVIDIA container runtime.
- Docker permission for the robot user.
- PICO / XRoboToolkit for live teleoperation.

The robot configuration contains 12 leg joints, 3 waist joints, and 14 arm
joints. Remove dexterous hands before running the policy unless your setup
explicitly includes and has validated them.

## First-Time Robot Setup

Do this once on the robot before the first deployment.

### Configure Docker Runtime

Confirm that the NVIDIA runtime is available:

```bash
docker info | grep -i runtime
```

Use `sudo` if the robot user does not yet have Docker permission. If `nvidia`
is missing, configure `/etc/docker/daemon.json`:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

Restart Docker only after stopping any running robot controller and confirming
that the robot is safe:

```bash
sudo systemctl restart docker
```

If necessary, add the robot user to the Docker group and re-login:

```bash
sudo usermod -aG docker "$USER"
```

### Check Network Interface

The container uses host networking. Find the interface connected to the robot
controller, which may differ from the Wi-Fi interface used by PICO:

```bash
ip addr
```

The image defaults to `eth0`, so no network override is needed for that setup.
Only if your robot uses another interface, add
`--set robot.network_interface=<your-interface>` to the teleoperation command.
Do not edit ROS launch files.

## Deployment Image

The standard image name is:

```text
horizonrobotics/holomotion:v2.0.0-orin-jp5.1-arm64
```

Use the supplied image archive for offline delivery:

```bash
docker load -i holomotion_v2.0.0_orin_jp5.1_arm64.tar
```

If the image is available in your registry, it can instead be pulled:

```bash
docker pull horizonrobotics/holomotion:v2.0.0-orin-jp5.1-arm64
```

The following steps assume the image is already loaded or pulled. TensorRT
engines are tied to the packaged model, hardware, and runtime versions; do not
reuse an unrelated engine cache or overlay old deployment code.

## Start Container

Run this on the robot:

```bash
docker run --rm -it \
  --runtime nvidia \
  --gpus all \
  --privileged \
  --network host \
  -v /dev:/dev \
  --name holomotion_g1 \
  --entrypoint /bin/bash \
  horizonrobotics/holomotion:v2.0.0-orin-jp5.1-arm64
```

All commands below run inside the container. No manual Conda activation or ROS
setup is needed. Run only one robot-control container at a time. If
`holomotion_g1` already exists, inspect it with `docker ps -a` rather than
automatically removing a possibly active controller.

## No-Action Check

```bash
holomotion check
```

The check validates the packaged model/configuration contract, robot assets,
and ONNX Runtime CUDA/TensorRT provider availability. It does not start ROS,
PICO, or a robot action publisher, and it does not load GPU inference sessions
or measure the full control-loop latency.

Continue only if the command exits successfully and reports both model
preflight and provider checks as passed. Any setup failure is a stop condition;
do not repeatedly restart a failing controller.

For file/import checks only, `holomotion check --structure-only` is available;
it is not a substitute for checking accelerator readiness on the robot.

## Robot Preparation

Before starting teleoperation:

1. Hang the robot or use a safe test fixture for the first run.
2. Remove dexterous hands if they are not part of the tested setup.
3. Power on the robot and confirm the controller and network are ready.
4. Enter Unitree debug mode with `L2 + R2` if required by your robot setup.
5. Keep the operator beside the remote controller, ready to press `Select`.
6. Complete `holomotion check` before launching the control stack.

## Live Teleoperation

The control-critical path runs on the robot:

```text
PICO / XRoboToolkit -> HoloRetarget -> VR reference queue
                   -> GPU joint-to-link adapter -> canonical link reference
                   -> observation -> motion policy -> robot
```

The GPU forward-kinematics adapter converts the existing retargeted robot joint
reference into link poses. It adapts the VR input to the canonical multi-point
reference interface; that interface does not depend on PICO. Observation
construction and motion-policy inference also run on the GPU. The reference,
observation, and policy clocks remain at 50 Hz.

### Reference Interface and Input Adapters

The motion policy uses a source-independent reference interface, not a PICO or
VR-specific signal format. The reference supplies target link positions and
orientations over the model's time grid, together with validity information.
The observation builder applies the selected control-point mask and combines
the reference with robot-state feedback and previous actions.

The canonical device reference is defined by `DeviceLinkReference` in
`deployment/unitree_g1_ros2_29dof/src/humanoid_policy/link_reference.py`.
Positions are world-frame metres and orientations are WXYZ quaternions; link
ordering, future-frame offsets, and CUDA buffer layouts must match the packaged
model. The observation builder performs the model-specific encoding.

PICO VR is the currently validated **input adapter**, not a requirement of the
policy interface. Its retarget and GPU FK steps adapt the existing VR
application's joint references into that common representation. Other input
sources can implement the same contract without inheriting the PICO/retarget
pipeline.

For example, a velocity-command adapter could generate a pelvis target
trajectory from translational and yaw-rate commands, then feed it through
`Pelvis-1`. It would need defined coordinates, initial-pose alignment, and
current/future targets on the model's 50 Hz time grid. Raw velocity values are
not direct motion-policy inputs. Such a velocity-to-pelvis adapter is an
extension point, not a command provided by this image, and must be validated
before real-robot use. The separate velocity policy retained in the current
controller is distinct from this proposed adapter.

### PICO / XRoboToolkit Setup

Use PICO 4 Ultra with XRoboToolkit body tracking. The recommended setup uses one
headset, two controllers, and two motion trackers strapped to the ankles. Keep
PICO and the robot on the same low-latency Wi-Fi network.

One-time setup:

1. Install the XRoboToolkit PICO app and enable Developer Mode on PICO.
2. Pair the motion trackers and calibrate full-body tracking.

Before entering motion tracking:

1. Start the selected teleoperation command below so the robot-side service runs.
2. Set `PC Service` in XRoboToolkit to the robot IP reachable from PICO.
3. Confirm the service status is `WORKING`.
4. Enable `Head`, `Controller`, `Full body`, and `Send`.
5. Stand in a stable calibration pose and wait for reference-queue readiness.

The launch profile configures the robot-side service and reference source:

```yaml
runtime:
  pico_service_command: "/opt/apps/roboticsservice/runService.sh"
  pico_service_log: "/tmp/holomotion_pico_service.log"

policy:
  reference_source: "pico_local"
  enable_teleop_reference: true
```

### Select Control Points

The following two released modes use the same packaged v2.0 motion policy:

| Mode | Reference control points |
| --- | --- |
| `Pelvis-1` | Pelvis |
| `Keypoint-5` | Pelvis, both wrists, and both ankles |

Start 5-point control:

```bash
holomotion teleop --mode Keypoint-5
```

For the other modes, use one of these commands instead:

```bash
holomotion teleop --mode Pelvis-1
```

Pass `--mode` immediately after `teleop`, and choose it explicitly. Omitting it
selects `Keypoint-5`. The mode determines the reference mask, not the number
of robot joints controlled. To change modes, safely stop the current
teleoperation process and relaunch with the new selection; no online hot-switch
command is exposed here.

Other model masks are experimental and are not exposed by the released teleoperation CLI.

### Controller Sequence

Wait for `Dual policies loaded successfully` and the PICO reference queue to
be ready. Then use the existing controller sequence:

```text
Start: ZERO_TORQUE -> MOVE_TO_DEFAULT
A:     MOVE_TO_DEFAULT -> POLICY (velocity tracking)
B:     enter motion tracking with the selected reference mask
Y:     return to velocity tracking
Select: emergency stop
```

Teleoperation first confirms that the official Unitree motion controller has
released control. `holomotion control-check` queries its status without
releasing control. A failed handoff blocks startup.

The standing transition must finish before `A` can enter policy control.
If no finite policy target arrives for 100 ms, the controller enters latched
`POLICY_HOLD`, retaining measured joint positions with position PD enabled.
Press `Start` to return through the default-posture transition, then `A` to
resume after fresh targets arrive. `Select` remains the explicit emergency stop.
This hold does not guarantee dynamic balance.

Do not press `A` until the default posture and policy setup are ready. If the
reference queue is not ready, keep the robot safe, fix PICO streaming, and wait
before pressing `B` again. Do not bypass readiness gates.

Inspect recent service output without streaming a large log:

```bash
tail -n 30 /tmp/holomotion_pico_service.log
```

## Offline Replay and Optional Viewer

This deployment exposes live multi-point teleoperation. `holomotion offline`
does not support replaying the old v1.4 NPZ motion format and exits with an
unsupported-operation error. Do not use old offline instructions as a v2.0
preflight. Simulation evaluation is documented separately in
[Evaluate Motion Tracking](evaluate_motion_tracking.md).

The optional workstation viewer receives telemetry on port `6002` and is not
part of the control path:

```bash
python holomotion_teleop_mjviewer.py --uri tcp://<robot-ip>:6002
```

## Stop

Preferred stop order:

1. Press `Y` to return to velocity tracking when appropriate.
2. Press `Select` immediately if emergency stop is needed.
3. Once the robot is safe, press `Ctrl+C` in the container terminal.

If the process cannot be stopped from that terminal, an operator can run:

```bash
docker stop holomotion_g1
```

Confirm the robot is safe before restarting a controller. Do not run multiple
teleoperation processes or change Docker/system services while it is active.

## Included Models

The image includes the velocity policy and a v2.0 motion policy, together with
their configurations. The motion checkpoint is published as
[HoloMotion_motion_tracking_model_v2.0.0](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_motion_tracking_model_v2.0.0),
containing `config.yaml`, `link_command_modes.json`, and
`exported/model_10500.onnx`.

The source deployment configuration defaults to
`deployment/unitree_g1_ros2_29dof/src/models/HoloMotion_motion_tracking_model_v2.0.0/`.
Keep the complete bundle together when packaging it; a delivered image may
configure a different internal model path. Users of the self-contained image
do not need a separate checkpoint download.

Use the model configuration and mode table packaged with that ONNX. They define
the actor layout, link ordering, future-frame offsets, and action contract;
substituting only an ONNX file is not sufficient. This deployment does not
support a v1.4.1 motion model in place of the v2.0 model.

## Safety and Troubleshooting

This is a real-robot demonstration and evaluation system, not a production-grade
controller. Keep a test fixture and an operator ready to stop the robot, and
stop immediately on unexpected behavior.

- **Check fails:** do not start teleoperation; resolve the first failed stage.
- **Network-interface error:** inspect `ip addr` on the robot and pass the
  correct `robot.network_interface`; do not assume `eth1` exists.
- **CUDA/TensorRT unavailable:** verify NVIDIA Container Runtime and the image's
  JetPack compatibility.
- **Killed process / OOM:** stop the control stack safely, inspect other GPU
  users and kernel logs, and verify the matching engine cache. Do not repeatedly
  cold-build engines while controlling the robot.
- **Motion-entry failure:** check PICO calibration, streaming, and reference
  freshness before entering motion tracking.
- **Policy-clock overruns:** measure retarget, FK, observation, and inference
  separately. Do not lower the 50 Hz clock to hide a failed timing contract.
