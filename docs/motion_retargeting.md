# Motion Retargeting

Complete [environment setup](environment_setup.md), including the licensed body
models, and run from the repository root:

```bash
source train.env
```

This repository uses a two-stage data pipeline for motion tracking training:

```text
raw human datasets -> HoloSMPL H5 -> HoloRetarget robot H5
```

HoloSMPL stores canonical human motion. The training production pipeline calls
HoloRetarget and writes the minimal robot pose reference into HDF5.

## HoloSMPL Schema

Each HoloSMPL clip contains:

- `human_pose_aa [T,72]`: root orientation plus SMPL 23 body joints in axis-angle.
- `human_root_trans [T,3]`: canonical z-up root translation in meters.
- `human_shape_beta [B]`: clip-level shape beta, not frame-broadcast.
- `human_root_height [T,1]`: derived from root translation.
- `human_gravity_projection [T,3]`: derived from root orientation.
- `metadata`: JSON provenance and dataset fields.

Packed HoloSMPL H5 keeps frame-major arrays at the shard root and stores `human_shape_beta` under `clips/human_shape_beta [num_clips,B]`.

## Build HoloSMPL

Convert a raw dataset to canonical HoloSMPL NPZ:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holosmpl convert-canonical \
  --dataset <dataset_name> \
  --input-root <raw_dataset_root> \
  --output-root <canonical_root> \
  --target-fps 50 \
  --overwrite
```

Convert canonical clips to HoloSMPL NPZ:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holosmpl convert-formal-npz \
  --canonical-root <canonical_root> \
  --output-root <holosmpl_npz_root> \
  --overwrite
```

Pack HoloSMPL NPZ into HoloSMPL H5:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holosmpl pack-formal-h5 \
  --formal-npz-root <holosmpl_npz_root> \
  --output-root <holosmpl_h5_root> \
  --compression lzf \
  --overwrite
```

## Build Robot Training H5

Run HoloRetarget on HoloSMPL H5 and write the existing robot HDF5 v2 format:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holosmpl retarget-holoretarget-h5 \
  --holosmpl-h5-root <holosmpl_h5_root> \
  --output-root <robot_h5_root> \
  --compression lzf \
  --overwrite
```

The output H5 stores only the non-derived robot reference:

- `ref_root_pos [T,3]`
- `ref_root_rot [T,4]` in `xyzw` order
- `ref_dof_pos [T,29]`

Training and deployment derive joint velocity, root velocity, projected
gravity, and local-frame velocity through the shared motion-tracking
observation module.

The output directory includes a `manifest.json` for these robot HDF5 shards.
The v2.0 training recipe also requires `metadata.source_id` for every eligible
clip and matching source-sampling probabilities; retargeting preserves input
metadata but does not assign those source labels for you. See
[training data configuration](train_motion_tracking.md#11-configure-your-training-data).

Human-side HoloSMPL NPZ and compact robot HDF5 are different formats from the
full robot reference NPZ used for interactive sim2sim. Use the
[evaluation guide](evaluate_motion_tracking.md) to select the appropriate input.

## Smoke Validation

A quick schema smoke can be run from an existing canonical root:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holosmpl convert-formal-npz \
  --canonical-root <canonical_root> \
  --output-root /tmp/holosmpl_lafan1_formal_npz \
  --overwrite \
  --progress-interval 20

"$Train_CONDA_PREFIX/bin/python" -m holosmpl pack-formal-h5 \
  --formal-npz-root /tmp/holosmpl_lafan1_formal_npz \
  --output-root /tmp/holosmpl_lafan1_h5 \
  --compression lzf \
  --shard-target-clips 10 \
  --overwrite \
  --progress-interval 20
```

HoloRetarget robot H5 generation requires a runtime where Newton/Warp can see a CUDA device, matching the online deployment runtime.
