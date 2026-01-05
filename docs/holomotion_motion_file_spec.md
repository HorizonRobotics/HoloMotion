## HoloMotion NPZ Format — Keys and Values

This document lists the exact keys saved in a HoloMotion NPZ and their value types/shapes.

- Prefix policy
  - ref_*: reference motion (source-of-truth produced by preprocessing)
  - ft_ref_*: filtered reference motion (post-filtering; never overwrites ref_*)
  - robot_*: robot states (only present in offline evaluation exports)
  - Legacy (no prefix): kept only for backward-compat; new files prefer ref_*

- metadata
  - type: JSON string
  - fields:
    - motion_key: str
    - raw_motion_key: str
    - motion_fps: float
    - num_frames: int
    - wallclock_len: float  (seconds, approx (num_frames - 1) / motion_fps)
    - num_dofs: int
    - num_bodies: int
    - clip_length: int  (original clip length in frames)
    - valid_prefix_len: int  (contiguous valid frames from the start)

- dof_pos
  - dtype: float32
  - shape: [T, num_dofs]  (URDF joint order; ref or robot)
  - deprecated

- dof_vels
  - dtype: float32
  - shape: [T, num_dofs]  (URDF joint order; ref or robot)
  - deprecated

- global_translation
  - dtype: float32
  - shape: [T, num_bodies, 3]  (meters, world frame; ref or robot)
  - deprecated

- global_rotation_quat
  - dtype: float32
  - shape: [T, num_bodies, 4]  (quaternion XYZW, world frame; ref or robot)
  - deprecated

- global_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (m/s, world frame; ref or robot)
  - deprecated

- global_angular_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (rad/s, world frame; ref or robot)
  - deprecated

- ref_dof_pos
  - dtype: float32
  - shape: [T, num_dofs]  (URDF joint order; reference motion)

- ref_dof_vel
  - dtype: float32
  - shape: [T, num_dofs]  (URDF joint order; reference motion)

- ref_global_translation
  - dtype: float32
  - shape: [T, num_bodies, 3]  (meters, world frame; reference motion)

- ref_global_rotation_quat
  - dtype: float32
  - shape: [T, num_bodies, 4]  (quaternion XYZW, world frame; reference motion)

- ref_global_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (m/s, world frame; reference motion)

- ref_global_angular_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (rad/s, world frame; reference motion)


- ft_ref_dof_pos
  - dtype: float32
  - shape: [T, num_dofs]  (filtered reference motion)

- ft_ref_dof_vel
  - dtype: float32
  - shape: [T, num_dofs]  (derived from filtered positions)

- ft_ref_global_translation
  - dtype: float32
  - shape: [T, num_bodies, 3]  (filtered reference motion)

- ft_ref_global_rotation_quat
  - dtype: float32
  - shape: [T, num_bodies, 4]  (filtered, normalized XYZW)

- ft_ref_global_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (derived from filtered positions)

- ft_ref_global_angular_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (derived from filtered quaternions)


 - robot_dof_pos
  - dtype: float32
  - shape: [T, num_dofs]  (URDF joint order; robot)

- robot_dof_vel
  - dtype: float32
  - shape: [T, num_dofs]  (URDF joint order; robot)

- robot_global_translation
  - dtype: float32
  - shape: [T, num_bodies, 3]  (meters, world frame; robot)

- robot_global_rotation_quat
  - dtype: float32
  - shape: [T, num_bodies, 4]  (quaternion XYZW, world frame; robot)

- robot_global_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (m/s, world frame; robot)

- robot_global_angular_velocity
  - dtype: float32
  - shape: [T, num_bodies, 3]  (rad/s, world frame; robot)



Notes:
- T == num_frames from metadata.
- All arrays are float32.
- Interpretation rule:
  - If ref_* keys are present: non-ref keys represent robot states, ref_* represent reference motion.
  - If ref_* keys are absent: non-ref keys represent reference motion states.
  - If ft_ref_* present: they are filtered counterparts of ref_* and may be used for analysis/visualization; consumers should pick appropriate inputs. When the `ft_as_legacy` preprocessing stage is used, legacy unprefixed keys are overwritten with the filtered `ft_ref_*` arrays (legacy keys then equal filtered reference, not raw ref_*).
-
- Kinematic consistency checker (`not_for_commit/motion_utils/npz_checker.py`):
  - Assumes all reference states (`ref_*`) are generated from a single root
    trajectory (body 0 position/orientation from `ref_global_*` or legacy
    `global_*`) and `dof_pos` via FK using the same robot config as for
    NPZ generation.
  - Verifies that:
    - Metadata (`motion_fps`, `num_frames`, `num_dofs`, `num_bodies`) matches
      the leading dimensions of any present `ref_*`, legacy, and `robot_*`
      arrays.
    - All arrays stored under the keys listed above have dtype float32.
    - Alias pairs (`ref_*` vs legacy unprefixed keys) are identical when
      `ft_as_legacy` is not applied:
      `ref_dof_pos`↔`dof_pos`, `ref_dof_vel`↔`dof_vels`,
      `ref_global_translation`↔`global_translation`,
      `ref_global_rotation_quat`↔`global_rotation_quat`,
      `ref_global_velocity`↔`global_velocity`,
      `ref_global_angular_velocity`↔`global_angular_velocity`.
    - For any present `ref_*` arrays, recomputing FK from
      (`ref_dof_pos`/`dof_pos`, root position/orientation from
      `ref_global_*`/`global_*`, `motion_fps`) reproduces
      `ref_dof_pos`, `ref_dof_vel`, `ref_global_translation`,
      `ref_global_rotation_quat`, `ref_global_velocity`,
      `ref_global_angular_velocity` up to a small numeric tolerance.