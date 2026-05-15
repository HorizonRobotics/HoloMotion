# Holomotion Teleop

Single-process pipeline for:

`PICO / XRoboToolkit -> SMPL conversion -> GMR retargeting -> robot ZMQ`


## Prerequisites

Before setting up the Python environment, install XRoboToolkit PC Service manually.

1. On Ubuntu 22.04, download the XRoboToolkit PC Service `.deb` package, or build it from source.

```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```


## Environment Setup

```bash
cd /path/to/holomotion_teleop
bash setup_holomotion_teleop_x86_ubuntu2204.sh
```

This script will:

- create the Conda environment `holomotion_teleop`
- automatically clone and install `GMR` and `SMPLSim`
- install runtime dependencies such as `numpy==1.23.5`, `torch`, and `pyzmq`
- build and install `xrobotoolkit_sdk` from source


Optional environment variables:

```bash
ENV_NAME=holomotion_teleop
PYTHON_VERSION=3.10
INSTALL_APT_DEPS=auto
THIRD_PARTY_DIR=/path/to/third_party
GMR_SOURCE_DIR=/path/to/GMR
SMPLSIM_SOURCE_DIR=/path/to/SMPLSim
XRT_PYBIND_REPO_DIR=/path/to/XRoboToolkit-PC-Service-Pybind
```

- `INSTALL_APT_DEPS=auto`: only runs apt installation if required build tools are missing
- `INSTALL_APT_DEPS=0`: skip apt installation entirely if your machine already has the tools or apt is unusable
- `INSTALL_APT_DEPS=1`: force the apt installation step
- `THIRD_PARTY_DIR`: default directory used for auto-cloned third-party repositories
- `GMR_SOURCE_DIR` / `SMPLSIM_SOURCE_DIR`: point to external source checkouts; if omitted, the script auto-clones them


## Input and Output

### Input

The script reads raw body tracking data directly from `xrobotoolkit_sdk.get_body_joints_pose()`:

- shape: `(24, 7)`
- row format: `[x, y, z, qx, qy, qz, qw]`

### Output

The robot-side ZMQ payload contains `latest_obs` as `float32[65]`:

1. `dof_pos[29]`
2. `dof_vel[29]`
3. `root_pos[3]`
4. `root_rot_wxyz[4]`

Additional metadata is included in the same payload:

- `frame_index`
- `timestamp_realtime`
- `timestamp_monotonic`
- `timestamp_ns`
- `pico_dt`
- `pico_fps`

## Next Steps

Before running teleoperation on the real robot, make sure the operators are already familiar with the offline `.npz` motion-performance workflow and the robot's basic mode-switching behavior. Teleoperation should not be the first time the team tests motion-mode entry on hardware.


### Real Robot Workflow

Use the following checklist when running the teleoperation stack on the real robot.

#### 1. Hardware and Network

Required hardware:

- PICO 4 / PICO 4 Pro headset
- 2 PICO controllers
- 2 PICO motion trackers attached to the ankles
- One workstation running `holomotion_teleop_node.py`
- One robot computer running the policy / control stack
- A low-latency Wi-Fi network shared by the PICO headset and the workstation

Make sure the robot, the workstation and the PICO headset are on the same Wi-Fi network. Low network latency is important for stable teleoperation. The PICO-side setup steps below follow the XRoboToolkit / PICO workflow described in the [GR00T VR Teleop Setup (PICO)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html).

#### 2. Install and Configure PICO

1. Install the XRoboToolkit PICO app on the headset.
   - Enable Developer Mode on the headset.
   - Open the browser on PICO and download the XRoboToolkit PICO APK.
   - Install the APK from the downloads page and confirm it appears in the app library.
2. Pair the two PICO motion trackers.
   - Attach one tracker to each ankle.
   - Open the motion tracker settings on the headset.
   - Unpair any old trackers first, then pair both trackers again.
3. Calibrate the motion trackers on the headset.
   - Follow the standing calibration step.
   - Then look down at the foot trackers so the headset cameras can detect them.
4. Connect the headset to the workstation.
   - Confirm the headset and workstation are on the same Wi-Fi network.
   - Open the XRoboToolkit app on the headset.
   - Enter the workstation IP address into the PC Service field.
   - Verify the status shows a successful connection.
5. In XRoboToolkit, enable the required streaming options.
   - Enable `Head` and `Controller` tracking.
   - Set `Pico Motion Tracker` to `Full body`.
   - Enable the `Send` option for data/control streaming.

#### 3. Configure the Robot-Side Policy

Before starting the robot-side policy, update the robot config file:

`HoloMotion/deployment/unitree_g1_ros2_29dof/launch_profiles/orin_docker.yaml`

Recommended settings:

- `enable_teleop_reference: true`
- `latest_obs_zmq_uri: "tcp://<workstation-ip>:6001""`

Replace `<workstation-ip>` with the actual IP address of the workstation that runs `holomotion_teleop_node.py`.

This ensures the robot waits for live VR data before switching into motion mode and connects to the correct ZMQ publisher endpoint.

#### 4. Launch Order

Start the system in the following order:

1. Start the robot control / policy stack on the robot computer.
2. Wait until the control policy is fully initialized, then press `Start` to move the robot into the default pose.
3. Start XRoboToolkit on the PICO headset and confirm that body-tracking data is being streamed.
4. Start the teleoperation node on the workstation:

```bash
conda activate holomotion_teleop
cd /path/to/holomotion_teleop
python holomotion_teleop_node.py
```

If needed, pass explicit ZMQ arguments such as:

```bash
python holomotion_teleop_node.py \
  --robot-zmq-uri tcp://*:6001 \
  --robot-zmq-mode bind \
  --hz 50
```

5. After the robot-side policy is receiving live teleoperation data, perform the runtime mode sequence:
   - press `A` to enter walking / velocity mode
   - press `B` to enter teleoperation motion mode
   - press `Y` whenever you want to leave teleoperation and return to walking mode
## Optional Arguments

- `--robot-zmq-uri`: robot-side ZMQ endpoint for the 65D output
- `--robot-zmq-mode`: `bind` or `connect`
- `--hz`: main loop frequency / processing cap
- `--timing-log-every`: print average stage timing every N ticks
- `--save-obs-path`: save emitted 65D observations on exit as `.npy` or `.npz`

#### 5. Runtime Check

Before enabling motion on the robot:

- confirm XRoboToolkit PC Service is running
- confirm the PICO headset is connected to the workstation
- confirm `holomotion_teleop_node.py` is publishing ZMQ data
- confirm the robot-side policy is using the correct workstation IP in `latest_obs_zmq_uri`
- confirm the robot-side config keeps `enable_teleop_reference: true`
- confirm the team has already validated the offline `.npz` motion-performance pipeline before attempting live teleoperation

Once the ZMQ stream is stable, enable the robot policy and switch into motion mode.