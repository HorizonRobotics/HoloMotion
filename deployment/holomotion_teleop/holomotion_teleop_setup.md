# HoloMotion Teleop Tools

This directory contains workstation-side tools for visualization, fake Pico
tests, and the legacy workstation-retarget pipeline.

For HoloMotion v1.4.0 real-robot deployment, do not run the workstation retarget
node as the control path. The Unitree G1 Docker image runs Pico/XRoboToolkit
input, HoloRetarget, observation construction, and policy inference on the
robot's Orin. Use this command inside the robot Docker container instead:

```bash
holomotion teleop
```

The current robot-side control path is:

```text
PICO / XRoboToolkit -> Orin HoloRetarget -> observation -> policy -> robot
```

The legacy workstation pipeline is still useful for debugging and offline
visualization:

`PICO / XRoboToolkit body_poses[24,7] -> HoloRetarget -> reference_qpos[36] ZMQ`

The retarget implementation lives in the repository-level `holoretarget/`
package. Files under `deployment/` only handle XRoboToolkit input, ZMQ output,
fake stream tests, and visualization.

## Environment Setup

Install XRoboToolkit PC Service manually first on Ubuntu 22.04:

```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Then create the Python environment:

```bash
cd /path/to/holomotion/deployment/holomotion_teleop
bash setup_holomotion_teleop_x86_ubuntu2204.sh
```

The setup script will:

- create/update conda env `holomotion_teleop` with Python 3.12
- verify the packaged HoloRetarget assets under `holoretarget/assets`
- build and install `xrobotoolkit_sdk`
- install `numpy`, `pyzmq`, `mujoco`, `newton==1.0.0`, `warp-lang==1.12.0`
- install this repo in editable mode so `holoretarget` can be imported

After setup, `start_holoretarget.sh`, `view_holoretarget.sh`, and
`view_orin_reference.sh` locate the `holomotion_teleop` Conda environment
automatically. Set `HOLOMOTION_TELEOP_ENV_NAME` for a different environment
name or `HOLOMOTION_PY` for an explicit Python executable. Recordings default
to `${XDG_DATA_HOME:-$HOME/.local/share}/holomotion/recordings`; override this
with `HOLORETARGET_RECORD_ROOT` when needed.

## Legacy Workstation Retarget

This mode runs Pico input and HoloRetarget on a workstation, then publishes
`reference_qpos` on ZMQ port `6001`. It is not the v1.4.0 Unitree G1 deployment
path. Use it only for debugging, recording reference streams, or comparing the
old workstation-retarget flow.

```bash
conda activate holomotion_teleop
cd /path/to/holomotion/deployment/holomotion_teleop
python holomotion_teleop_node.py \
  --robot-zmq-uri tcp://*:6001 \
  --robot-zmq-mode bind \
  --hz 50 \
  --timing-log-every 250
```

Useful optional flags:

- `--asset-root`: override the HoloRetarget asset root; the default is
  `/path/to/holomotion/holoretarget/assets`
- `--save-reference-path`: save emitted reference qpos frames on exit
- `--debug-retarget-dump`: save input body poses and reference qpos for debugging
- `--skip-start-service`: do not auto-run `/opt/apps/roboticsservice/runService.sh`
  when debugging the workstation XRoboToolkit service manually

## Fake Pico Test

Use a recorded Pico CSV without wearing the headset:

```bash
python holomotion_teleop_node.py \
  --fake-benchmark \
  --fake-pico-csv /path/to/pico_raw.csv \
  --fake-warmup 30 \
  --fake-frames 200 \
  --hz 50
```

To publish a fake stream over ZMQ:

```bash
python holomotion_teleop_node.py \
  --fake-pico-stream \
  --fake-pico-csv /path/to/pico_raw.csv \
  --fake-stream-frames 1000 \
  --robot-zmq-uri tcp://*:6001 \
  --robot-zmq-mode bind \
  --hz 50
```

This ZMQ fake stream targets the legacy workstation-retarget interface on port
`6001`; it is not required for the v1.4.0 robot Docker teleop flow.

## Viewer

With robot-side local Retarget, connect the viewer to the Orin telemetry port:

```bash
python holomotion_teleop_mjviewer.py --uri tcp://<robot-ip>:6002
```

Port `6002` is best-effort visualization telemetry. Missing frames are expected
and never feed back into robot control.

For the legacy workstation-Retarget mode, use port `6001`:

```bash
conda activate holomotion_teleop
cd /path/to/holomotion/deployment/holomotion_teleop
python holomotion_teleop_mjviewer.py --uri tcp://127.0.0.1:6001
```

The viewer uses the packaged MJCF at `holoretarget/assets/unitree_g1/g1_mocap_29dof.xml`.
Use `--mjcf` only when debugging a different robot asset.

Use `--dry-run` for a connectivity check without opening MuJoCo.

## Output Contract

The ZMQ payload contains `reference_qpos: float32[36]`:

1. `root_pos[3]`
2. `root_rot_wxyz[4]`
3. `dof_pos[29]`

The policy observation layer derives velocity, projected gravity, and local
root velocity from this reference stream.

Metadata fields include frame index, realtime/monotonic timestamps, Pico device
timestamp, Pico dt, and Pico fps.
