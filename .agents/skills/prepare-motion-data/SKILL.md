---
name: prepare-motion-data
description: Convert a supported or custom human-motion source into HoloSMPL and HoloRetarget data for HoloMotion training. Use when users bring their own BVH, SMPL/SMPL-X, device export, video-derived motion, or another motion format and need conversion, validation, retargeting, or a new source adapter.
---

# Prepare Motion Data

Turn user motion into validated HoloMotion training data:

```text
raw source -> canonical HoloSMPL -> formal HoloSMPL -> robot HDF5
```

Read `holosmpl/README.md` and `docs/motion_retargeting.md` before choosing commands. Use `docs/holomotion_motion_file_spec.md` when producing or inspecting deployment/evaluation NPZ files.

## Determine the input contract

1. Identify the source format, coordinate frame, units, frame rate, skeleton, joint order, and whether shape parameters are available.
2. Run `python -m holosmpl list-sources` and inspect the matching README under `holosmpl/supported_datasets/` or `holosmpl/supported_devices/`.
3. Do not describe an arbitrary format as supported merely because it resembles SMPL, BVH, or another registered source.
4. For an unsupported format, inspect representative files and add a source adapter following the `Adding a Source` section in `holosmpl/README.md`.

Keep raw user data outside Git. Do not commit motion datasets, generated HDF5/NPZ files, licensed body models, or personal paths.

## Convert and validate

For a registered source, prefer the end-to-end entry point:

```bash
python -m holosmpl convert \
  --source <source_name> \
  --input-root <raw_root> \
  --output-root <output_root>
```

Use the source-specific README for required options. Start with a small representative subset before converting a complete dataset.

Validate the canonical result before retargeting:

- Z-up world frame and meter scale;
- expected pose layout and body orientation;
- stable root translation and floor/contact behavior;
- correct 50 Hz output;
- finite arrays, consistent frame counts, and useful provenance metadata.

Use HoloSMPL visualization or rendering commands from `holosmpl/README.md`. Do not accept schema validity as proof that the motion semantics are correct.

## Retarget for training

Use the formal HoloSMPL H5 output as the HoloRetarget input. Follow `docs/motion_retargeting.md` to produce robot HDF5.

HoloRetarget requires a Newton/Warp runtime with a visible CUDA device. Use the project training environment and validate a small shard before scaling.

Check that the resulting robot data has the documented reference arrays, 29-DOF ordering for the supported G1 pipeline, finite values, plausible joint limits, and correct clip metadata.

## Completion

Report:

- source type and adapter used;
- input assumptions;
- output roots and formats;
- schema and visual checks performed;
- rejected or suspicious clips;
- whether robot HDF5 was produced;
- remaining unsupported semantics.

Do not claim the dataset is training-ready until both human-motion validation and robot-retarget validation pass.
