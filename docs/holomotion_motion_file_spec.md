# HoloMotion Motion File Formats

This document describes retargeted NPZ files and the compact root/DOF HDF5
format shared by training and evaluation. For dataset configuration and
sampling, see [training](train_motion_tracking.md). For clip selection,
evaluation commands and metrics, see [evaluation](evaluate_motion_tracking.md).

## NPZ — Keys and Values

This document lists the exact keys saved in a HoloMotion NPZ and their value types/shapes.

- Prefix policy

  - ref\_\*: reference motion (source-of-truth produced by preprocessing)
  - ft*ref*_: filtered reference motion (post-filtering; never overwrites ref\__)
  - robot\_\*: robot states (only present in offline evaluation exports)
  - Legacy (no prefix): kept only for backward-compat; new files prefer ref\_\*

- metadata

  - type: JSON string
  - fields:
    - motion_key: str
    - raw_motion_key: str
    - motion_fps: float
    - num_frames: int
    - wallclock_len: float (seconds, approx (num_frames - 1) / motion_fps)
    - num_dofs: int
    - num_bodies: int
    - clip_length: int (original clip length in frames)
    - valid_prefix_len: int (contiguous valid frames from the start)

- ref_dof_pos

  - dtype: float32
  - shape: [T, num_dofs] (URDF joint order; reference motion)

- ref_dof_vel

  - dtype: float32
  - shape: [T, num_dofs] (URDF joint order; reference motion)

- ref_global_translation

  - dtype: float32
  - shape: [T, num_bodies, 3] (meters, world frame; reference motion)

- ref_global_rotation_quat

  - dtype: float32
  - shape: [T, num_bodies, 4] (quaternion XYZW, world frame; reference motion)

- ref_global_velocity

  - dtype: float32
  - shape: [T, num_bodies, 3] (m/s, world frame; reference motion)

- ref_global_angular_velocity

  - dtype: float32
  - shape: [T, num_bodies, 3] (rad/s, world frame; reference motion)

- ft_ref_dof_pos

  - dtype: float32
  - shape: [T, num_dofs] (filtered reference motion)

- ft_ref_dof_vel

  - dtype: float32
  - shape: [T, num_dofs] (derived from filtered positions)

- ft_ref_global_translation

  - dtype: float32
  - shape: [T, num_bodies, 3] (filtered reference motion)

- ft_ref_global_rotation_quat

  - dtype: float32
  - shape: [T, num_bodies, 4] (filtered, normalized XYZW)

- ft_ref_global_velocity

  - dtype: float32
  - shape: [T, num_bodies, 3] (derived from filtered positions)

- ft_ref_global_angular_velocity

  - dtype: float32
  - shape: [T, num_bodies, 3] (derived from filtered quaternions)

- robot_dof_pos
  - dtype: float32
  - shape: [T, num_dofs] (URDF joint order; robot)

- robot_dof_vel

  - dtype: float32
  - shape: [T, num_dofs] (URDF joint order; robot)

- robot_global_translation

  - dtype: float32
  - shape: [T, num_bodies, 3] (meters, world frame; robot)

- robot_global_rotation_quat

  - dtype: float32
  - shape: [T, num_bodies, 4] (quaternion XYZW, world frame; robot)

- robot_global_velocity

  - dtype: float32
  - shape: [T, num_bodies, 3] (m/s, world frame; robot)

- robot_global_angular_velocity

  - dtype: float32
  - shape: [T, num_bodies, 3] (rad/s, world frame; robot)

- dof_pos (deprecated legacy key)

  - dtype: float32
  - shape: [T, num_dofs] (URDF joint order; ref or robot)
  - deprecated

- dof_vels (deprecated legacy key)

  - dtype: float32
  - shape: [T, num_dofs] (URDF joint order; ref or robot)
  - deprecated

- global_translation (deprecated legacy key)

  - dtype: float32
  - shape: [T, num_bodies, 3] (meters, world frame; ref or robot)
  - deprecated

- global_rotation_quat (deprecated legacy key)

  - dtype: float32
  - shape: [T, num_bodies, 4] (quaternion XYZW, world frame; ref or robot)
  - deprecated

- global_velocity (deprecated legacy key)

  - dtype: float32
  - shape: [T, num_bodies, 3] (m/s, world frame; ref or robot)
  - deprecated

- global_angular_velocity (deprecated legacy key)
  - dtype: float32
  - shape: [T, num_bodies, 3] (rad/s, world frame; ref or robot)
  - deprecated

Notes:

- T == num_frames from metadata.
- All arrays are float32.

## Compact root/DOF HDF5

The `hdf5_v2` backend reads a manifest and HDF5 shards. This is a storage
format, not a fixed benchmark dataset or a data package included in the
release. A user dataset can be organized as:

```text
<my_dataset>/
├── manifest.json
└── shards/*.h5
```

Separate training and validation directories each have their own manifest.
Evaluation can use the same format with a separate, explicit clip list;
selection-file fields are described in the [evaluation guide](evaluate_motion_tracking.md#3-quantitative-evaluation-on-your-hdf5-dataset).

### Manifest

`manifest.json` indexes clip names and their stored frame ranges:

- `hdf5_shards[].file`: shard filename. Relative paths resolve from the
  manifest directory; absolute paths can reference shards elsewhere without
  copying them alongside the manifest.
- `clips[key].shard`: index into `hdf5_shards`.
- `clips[key].start` and `length`: starting frame in the shard and clip length.
- `clips[key].metadata.motion_fps`: reference frame rate; the loader also
  accepts `clips[key].motion_fps`.
- `dof_names`: DOF order, which must match the robot and model contract.
- `clips[key].metadata.source_id`: stable data-source label used by the
  default adaptive training sampler. Evaluation does not require source ratios.

Clip names must be unique across loaded manifests. Each referenced shard must
exist, each clip's frame range must fit that shard, and FPS must be positive.
Provenance fields such as `root` or `source_npz` do not replace the shard path.

### Shard arrays

The required arrays are stored as float32 with frames along the first axis:

| Key | Shape | Meaning |
| --- | --- | --- |
| `ref_root_pos` | `[T, 3]` | Root position in meters, world frame |
| `ref_root_rot` | `[T, 4]` | Root quaternion in XYZW order, world frame |
| `ref_dof_pos` | `[T, num_dofs]` | Joint positions in the declared DOF order |

`T` is the number of stored frames in the shard; a clip uses its indexed
slice. An optional integer `frame_flag` has one value per stored frame.
`Hdf5RootDofDataset` reads the indexed slices and computes body states through
FK and normalization in memory; evaluation does not create per-clip NPZ input
copies. This compact format differs from the older full-state `hdf5` backend.
