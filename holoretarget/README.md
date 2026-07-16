# HoloRetarget

HoloRetarget is the shared retargeting core used by online teleoperation and
training reference production. Its output is `reference_qpos[36]`:
`root_pos[3] + root_quat_wxyz[4] + dof_pos[29]`. Derived policy observations,
dataset I/O, and distributed scheduling live outside this package.

## Minimal API

HoloRetarget requires a CUDA-capable Newton/Warp runtime. The online API accepts
XRoboToolkit/Pico body poses directly and does not require an SMPL model. Input
is `float32[24,7]`, with position in the first three values and a Unity-coordinate
`xyzw` quaternion in the last four values:

```python
import numpy as np

from holoretarget import HoloRetargeter

retargeter = HoloRetargeter()
body_poses = np.zeros((24, 7), dtype=np.float32)
body_poses[:, 6] = 1.0  # Identity quaternion in xyzw order.
reference_qpos = retargeter.retarget_qpos_from_body_poses(body_poses)
assert reference_qpos.shape == (36,)
```

Call `reset_sequence()` before starting a new independent motion sequence. For
offline robot-HDF5 production, use `python -m holosmpl
retarget-holoretarget-h5`. For live teleoperation, see
`deployment/holomotion_teleop/` and `docs/realworld_deployment.md`.

## Runtime Asset Layout

HoloRetarget keeps runtime assets intentionally small and self-contained:

```text
holoretarget/assets/
  target_configs/smplx_to_g1.json
  unitree_g1/g1_mocap_29dof.xml
  unitree_g1/meshes/*.STL
```

The target config is maintained by HoloRetarget. Third-party robot description
assets are listed under `holoretarget/assets/THIRD_PARTY_NOTICES.md`.

The licensed SMPL model is only needed when converting SMPL-family training
data into the 24-joint body representation. It is not needed for Pico
teleoperation. Training conversion expects the neutral model at:

```text
thirdparties/smpl_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
```

Download `SMPL_python_v.1.1.0.zip` from the official SMPL website and extract it
under `thirdparties/smpl_models/`. The directory is intentionally gitignored.

For private cluster jobs, `scripts/orchard_holoretarget_submit.sh` validates this
file and copies the dereferenced model into the private job package. Do not
commit the SMPL model file or include it in public release archives.
