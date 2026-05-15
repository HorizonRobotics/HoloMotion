# HoloMotion Real-World Deployment Guide

This guide describes the Docker-only workflow for deploying HoloMotion on a physical Unitree G1 robot with 29 DOF.

Laptop / local deployment is no longer supported in this release. All robot-side code should run inside the Orin Docker container on the robot.

## Quick Reference

- Robot platform: Unitree G1 onboard NVIDIA Jetson Orin
- Docker container: `holomotion_orin_deploy`
- Docker image: `horizonrobotics/holomotion:orin_foxy_jp5.1_humble_deploy_zmq_full_20260509`
- Container repo path: `/home/unitree/holomotion`
- Deployment path: `/home/unitree/holomotion/deployment/unitree_g1_ros2_29dof`
- Launch profile: `deployment/unitree_g1_ros2_29dof/launch_profiles/orin_docker.yaml`
- Robot config: `deployment/unitree_g1_ros2_29dof/src/config/g1_29dof_holomotion.yaml`

The 29 DOF robot configuration contains 12 leg joints, 3 waist joints, and 14 arm joints.

For safety, remove the dexterous hands before running the policy.

## First-Time Robot Setup

Do this once on the robot before the first deployment.

### Configure Docker Runtime

Docker must be installed with NVIDIA Container Runtime support. Confirm the NVIDIA runtime is available:

```bash
sudo docker info | grep -i runtime
```

If the NVIDIA runtime is missing, configure `/etc/docker/daemon.json`:

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

If Docker permission fails, add the current user to the Docker group and re-login:

```bash
sudo usermod -aG docker $USER
groups
```

### Prepare Docker Image

Pull the official image:

```bash
docker pull horizonrobotics/holomotion:orin_foxy_jp5.1_humble_deploy_zmq_full_20260509
```


### Check Network Interface

On the robot, check the network interface used by Unitree ROS 2:

```bash
ifconfig
```

The interface is usually `eth0`. If it is different, update `robot.network_interface` in `deployment/unitree_g1_ros2_29dof/launch_profiles/orin_docker.yaml`.

Do not edit the ROS launch file for network configuration.

## Before Running

### Required Robot-Side Files

Check the model and motion data paths in:

`deployment/unitree_g1_ros2_29dof/src/config/g1_29dof_holomotion.yaml`

```yaml
velocity_tracking_model_folder: "velocity_tracking_model"
motion_tracking_model_folder: "<your_motion_model_folder>"
motion_clip_dir: "motion_data"
```

Expected model folder structure:

```bash
deployment/unitree_g1_ros2_29dof/src/models/<your_model_folder>
├── config.yaml
└── exported
    └── <model>.onnx
```

### Pre-trained Models

We provide pre-trained models that you can download and use:

- Motion Tracking Model: [Hugging Face](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_motion_tracking_model)
- Velocity Tracking Model: [Hugging Face](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_velocity_tracking_model)

To use these models:

1. Download the `HoloMotion_motion_tracking_model` and `HoloMotion_velocity_tracking_model` folders from the Hugging Face repository.
2. Place the downloaded folders under `deployment/unitree_g1_ros2_29dof/src/models/`, for example:

```bash
deployment/unitree_g1_ros2_29dof/src/models/
├── HoloMotion_motion_tracking_model/
└── HoloMotion_velocity_tracking_model/
```

3. Update `motion_tracking_model_folder` and `velocity_tracking_model_folder` in `g1_29dof_holomotion.yaml` to point to these folders.

Expected offline motion data:

```bash
deployment/unitree_g1_ros2_29dof/src/motion_data/
└── <retargeted_motion>.npz
```

### Launch Profile

Edit:

`deployment/unitree_g1_ros2_29dof/launch_profiles/orin_docker.yaml`

Common settings:

```yaml
robot:
  network_interface: "eth0"

policy:
  inference_backend: "tensorrt"
  enable_teleop_reference: false
  latest_obs_zmq_uri: "tcp://<workstation-ip>:6001"
  latest_obs_zmq_topic: "obs65"
  latest_obs_zmq_mode: "connect"
  latest_obs_zmq_conflate: true
  zmq_jitter_delay_frames: 0
  max_data_age: 0.6
  timing_debug_enabled: false
```

## Start Docker

Run this on the robot before each deployment session:

```bash
cd <your_holomotion_repo_path>/deployment/unitree_g1_ros2_29dof
bash start_container.sh
```

When prompted, enter the full local repository path, for example:

```bash
/home/unitree/HoloMotion
```

The script starts a fresh `holomotion_orin_deploy` container and mounts the repository to `/home/unitree/holomotion`.

Inside the container:

```bash
cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof
```

## Robot Preparation

Do this before launching the policy controller:

1. Put the robot in a hanging position.
2. Power on the robot and wait for zero torque mode.
3. Confirm the robot network is ready.
4. Enter debug mode with `L2 + R2` if needed. New deployments may enter this mode automatically.
5. Keep the operator ready to press `Select` for emergency stop.

## Offline Motion

Use this mode when motion tracking should execute local `.npz` clips.

### Configure

Set:

```yaml
policy:
  enable_teleop_reference: false
```

Check:

- `motion_tracking_model_folder` points to the intended motion model.
- `velocity_tracking_model_folder` points to the intended walking model.
- `motion_clip_dir` contains retargeted `.npz` files.
- `inference_backend` is `tensorrt`.

### Launch

Inside Docker:

```bash
cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof
bash launch_holomotion_29dof_docker.sh
```

### Controller Sequence

1. Press `Start` to move to the default pose.
2. Lower the robot to the ground when the default pose is stable.
3. Press `A` to enter velocity tracking.
4. Select a motion clip with the D-pad:
   - `Left`: first clip
   - `Right`: last clip
   - `Up`: previous clip
   - `Down`: next clip
5. Press `B` to enter motion tracking and execute the selected clip.
6. After the offline clip finishes, the controller automatically returns to velocity tracking.

You can still press `Y` during motion tracking to return to velocity tracking before the clip finishes.

## Teleoperation

Use this mode when motion tracking should follow live VR / teleoperation data.

Robot-side data flow:

```text
PICO / XRoboToolkit -> workstation teleop node -> ZMQ latest_obs -> robot policy node
```

For workstation setup, see:

`deployment/holomotion_teleop/holomotion_teleop_setup.md`

### Configure

Set:

```yaml
policy:
  enable_teleop_reference: true
  latest_obs_zmq_uri: "tcp://<workstation-ip>:6001"
  latest_obs_zmq_topic: "obs65"
  latest_obs_zmq_mode: "connect"
  latest_obs_zmq_conflate: true
  zmq_jitter_delay_frames: 0
  max_data_age: 0.6
```

Replace `<workstation-ip>` with the workstation IP that publishes `latest_obs`.

Check:

- XRoboToolkit PC Service is running on the workstation.
- PICO headset, controllers, and ankle trackers are connected and streaming.
- `holomotion_teleop_node.py` is publishing `latest_obs`.
- Robot-side `latest_obs_zmq_uri` points to the workstation IP.
- Offline motion has already been validated before live teleoperation.

### Launch Order

1. Start the Docker container on the robot.
2. Launch the robot policy stack inside Docker:

```bash
cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof
bash launch_holomotion_29dof_docker.sh
```

3. Press `Start` to move the robot to the default pose.
4. Start XRoboToolkit on the PICO headset and confirm tracking is active.
5. Start the teleoperation node on the workstation:

```bash
conda activate holomotion_teleop
cd /path/to/holomotion_teleop
python holomotion_teleop_node.py \
  --robot-zmq-uri tcp://*:6001 \
  --robot-zmq-mode bind \
  --hz 50
```

6. Wait for robot-side logs showing ZMQ data is ready.
7. Press `A` to enter velocity tracking.
8. Press `B` to enter teleoperation motion tracking.
9. Press `Y` to return to velocity tracking.

### ZMQ Notes

- `latest_obs_zmq_conflate: true` keeps the newest ZMQ packet and avoids backlog.
- `zmq_jitter_delay_frames: 0` uses the newest frame and gives the lowest latency.
- `zmq_jitter_delay_frames: 1` adds about one policy frame of delay, usually about 20 ms at 50 Hz, and can be more stable if the stream jitters.
- `max_data_age` controls stale data detection. If live data is too old, motion mode switches back to velocity mode for safety.

## Controller Reference

| State / Mode | Operation |
| --- | --- |
| Zero torque | Startup state. Joints should be loose. |
| Move to default | Press `Start`. |
| Velocity tracking | Press `A` after the robot reaches default pose. |
| Motion tracking | Press `B` from velocity tracking. Uses offline `.npz` or live teleop depending on `enable_teleop_reference`. Offline `.npz` clips automatically return to velocity tracking after completion. |
| Return to velocity | Press `Y` from motion tracking. |
| Select motion clip | In velocity tracking, use D-pad: `Left` first, `Right` last, `Up` previous, `Down` next. |
| Emergency stop | Press `Select`. |

## Logs

Useful logs:

- `[Timing-Agg]`: policy loop timing summary.
- `onnx_ms`: ONNX Runtime / TensorRT inference time.
- `policy_total_ms`: total policy step time.
- `[VR-STATUS]`: whether live ZMQ `latest_obs` is being received.

If `enable_teleop_reference=false`, `[VR-STATUS] No new ZMQ latest_obs` can be ignored.

## Recording

Use `--record` to enable rosbag recording:

```bash
bash launch_holomotion_29dof_docker.sh --record
```

Default topics:

- `/lowcmd`
- `/lowstate`
- `/humanoid/action`

## Stop

To stop control:

- Press `Select` on the remote controller for emergency stop, or
- Press `Ctrl+C` in the Docker terminal.

After stopping, confirm the robot is safe before restarting the controller.

## Safety

This deployment is for real-robot demonstration and evaluation. It is not a production-grade control system.

- Keep the robot in a hanging position during startup.
- Keep an operator near the remote controller.
- Do not stand close to the robot during motion tracking.
- Validate offline motion before live teleoperation.
- Stop immediately if the robot behaves unexpectedly.
