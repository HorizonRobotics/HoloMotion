# HoloMotion Real-World Deployment

This guide describes the Docker workflow for deploying HoloMotion v1.4.0 on a
Unitree G1 29-DOF robot with onboard NVIDIA Jetson Orin.

The v1.4.0 deployment image is self-contained: it includes the robot-side
deployment code, velocity model, motion-tracking model, and one filtered offline
motion sample. Users do not need to clone the repository or download SMPL models
for robot deployment.

## Requirements

- Unitree G1 29-DOF robot with Jetson Orin.
- JetPack 5.1 compatible NVIDIA container runtime.
- Docker permission for the robot user.
- PICO / XRoboToolkit only for live teleoperation.

The 29-DOF robot configuration contains 12 leg joints, 3 waist joints, and 14
arm joints. For safety, remove dexterous hands before running the policy unless
your deployment setup explicitly includes them.

## First-Time Robot Setup

Do this once on the robot before the first deployment.

### Configure Docker Runtime

Docker must be installed with NVIDIA Container Runtime support. Confirm the
NVIDIA runtime is available:

```bash
docker info | grep -i runtime
```

If the command needs administrator permission on your robot, run:

```bash
sudo docker info | grep -i runtime
```

If `nvidia` is missing, configure `/etc/docker/daemon.json`:

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

Restart Docker:

```bash
sudo systemctl restart docker
```

If Docker permission fails for the robot user, add the user to the Docker group
and re-login:

```bash
sudo usermod -aG docker $USER
groups
```

### Check Network Interface

HoloMotion uses host networking inside Docker. Confirm the Unitree ROS 2 network
interface is available before starting the container:

```bash
ip addr
```

The interface is commonly `eth0` on the robot. If your setup uses a different
interface, update the robot launch profile in your deployment image or derived
container. Do not edit ROS launch files directly for network configuration.

## Pull Image

```bash
docker pull horizonrobotics/holomotion:v1.4.0-orin-jp5.1-arm64
```

For an offline transfer, load the exported tar instead:

```bash
docker load -i holomotion_v1.4.0_orin_jp5.1_arm64.tar
```

## Start Container

Run this on the robot:

```bash
docker run --rm -it \
  --runtime nvidia \
  --gpus all \
  --privileged \
  --network host \
  --name holomotion_g1 \
  --entrypoint bash \
  horizonrobotics/holomotion:v1.4.0-orin-jp5.1-arm64
```

All commands below are executed inside the container.

## Command Summary

Recommended first test order:

```bash
holomotion check
holomotion offline
holomotion teleop
```

`holomotion check` is a no-action validation command. `holomotion offline` and
`holomotion teleop` start the robot control stack and should only be run after
the no-action check passes.

## No-Action Check

This command validates files, runtime setup, ONNX Runtime providers, and the
launch profile. It does not send actions to the robot.

```bash
holomotion check
```

Continue only if the final line is:

```text
HoloMotion Docker check PASSED. No robot action was sent.
```

## Robot Preparation

Before launching `offline` or `teleop`:

1. Hang the robot or keep it in a safe test fixture for the first run.
2. Remove dexterous hands if they are not part of the tested setup.
3. Power on the robot and wait for zero-torque / safe startup state.
4. Confirm the robot network and controller connection are ready.
5. Enter Unitree debug mode with `L2 + R2` if your robot setup requires it.
6. Keep the operator near the remote controller and ready to press `Select` for
   emergency stop.

Validate `holomotion check` before running any command that can send actions.
Validate offline motion before live teleoperation.

## Offline Motion Tracking

The image includes:

```text
qinghai_v1_4_0.npz
```

Start offline motion tracking:

```bash
holomotion offline
```

Remote controller sequence:

```text
A: enter policy / move-to-default
B: switch to motion tracking and execute the offline motion
Y: return to velocity mode
Select: emergency stop
```

To run your own `.npz`, mount a host directory when starting the container:

```bash
docker run --rm -it \
  --runtime nvidia \
  --gpus all \
  --privileged \
  --network host \
  -v /home/unitree/offline_data:/data:ro \
  --name holomotion_g1 \
  --entrypoint bash \
  horizonrobotics/holomotion:v1.4.0-orin-jp5.1-arm64
```

Then run:

```bash
holomotion offline /data/your_motion.npz
```

The `.npz` must follow the HoloMotion v1.4 motion format with these arrays:

```text
ref_dof_pos
ref_dof_vel
ref_global_translation
ref_global_rotation_quat
ref_global_velocity
ref_global_angular_velocity
```

## Live Teleoperation

Teleoperation runs the latency-sensitive path entirely on the robot:

```text
PICO / XRoboToolkit -> Orin HoloRetarget -> observation -> policy -> robot
```

The optional workstation viewer receives telemetry from the robot on port
`6002`; it is not part of the control path.

### PICO / XRoboToolkit Setup

Use PICO 4 Ultra with XRoboToolkit body tracking. The recommended
setup uses one headset, two controllers, and two PICO motion trackers strapped
to the ankles. Keep PICO and the robot on the same low-latency Wi-Fi network.

One-time setup:

1. Install the XRoboToolkit PICO app and enable Developer Mode on PICO.
2. Pair the two PICO motion trackers and calibrate full-body tracking.

Before teleoperation:

1. Start `holomotion teleop` so the robot-side XRoboToolkit service is running.
2. In the XRoboToolkit PICO app, set `PC Service` to the robot IP address.
3. Confirm the status is `WORKING`.
4. Enable `Head`, `Controller`, `Full body` and `Send`.
5. Stand in a stable calibration pose until body tracking is visible.

The robot-side service command is configured in the launch profile:

```yaml
runtime:
  pico_service_command: "/opt/apps/roboticsservice/runService.sh"
  pico_service_log: "/tmp/holomotion_pico_service.log"

policy:
  reference_source: "pico_local"
  enable_teleop_reference: true
```

If your image or robot uses a different XRoboToolkit service path, update the
launch profile in the image or in your derived container. Keep
`reference_source: "pico_local"` for the v1.4.0 on-robot teleoperation path.

Start teleoperation:

```bash
holomotion teleop
```

Remote controller sequence:

```text
A: enter policy / move-to-default
B: switch to HoloRetarget teleoperation motion tracking
Y: return to velocity mode
Select: emergency stop
```

If the logs say the VR queue is not ready, wait until PICO / XRoboToolkit data
is streaming, then press `B` again.

Useful checks inside the container:

```bash
tail -f /tmp/holomotion_pico_service.log
```

Expected startup logs include:

```text
Starting Pico service: /opt/apps/roboticsservice/runService.sh
Pico service ready
Reference source: local Pico/XRoboToolkit on the policy clock
```

HoloRetarget consumes PICO 24-joint global poses directly in v1.4.0. The robot
deployment image does not require an SMPL model.

## Optional Viewer

Run the viewer on a workstation that can reach the robot:

```bash
python holomotion_teleop_mjviewer.py \
  --uri tcp://<robot-ip>:6002
```

The viewer may drop frames or pause without affecting robot control.

## Stop

Preferred stop order:

1. Press `Y` to return to velocity mode if the robot is in motion tracking.
2. Press `Select` for emergency stop if needed.
3. Press `Ctrl+C` in the container terminal.

From another terminal:

```bash
docker stop holomotion_g1
```

## Safety

This deployment is for real-robot demonstration and evaluation. It is not a
production-grade control system.

- Keep the robot hanging or in a safe fixture during startup and first tests.
- Keep an operator near the remote controller at all times.
- Do not stand close to the robot during motion tracking or teleoperation.
- Run `holomotion check` before `offline` or `teleop`.
- Validate offline motion tracking before live teleoperation.
- Stop immediately if the robot behaves unexpectedly.
- Confirm the robot is safe before restarting the controller after any stop.

## Included Models

The Docker image includes the v1.4.0 release model:

```text
HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx
```

The standalone model package is available on Hugging Face:

- [HoloMotion motion tracking model v1.4.0](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_motion_tracking_model_v1.4.0)
- [HoloMotion velocity tracking model](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_velocity_tracking_model)

Use the standalone model package for training, finetuning, or custom deployment
packaging. Real-robot v1.4.0 users should start from the Docker image above.

## Troubleshooting

- If `holomotion check` fails, do not run `offline` or `teleop`.
- If ONNX Runtime does not list TensorRT/CUDA providers, check the NVIDIA Docker
  runtime.
- If teleoperation does not enter motion tracking, confirm PICO tracking and
  XRoboToolkit data streaming first.
- If a custom `.npz` fails to load, validate its six `ref_*` arrays and 29-DOF /
  30-body layout.
