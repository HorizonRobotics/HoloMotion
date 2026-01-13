## HoloMotion NPZ Format — Keys and Values

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
