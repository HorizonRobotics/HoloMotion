# MotionMillion

- Source type: public gated dataset
- Download: https://huggingface.co/datasets/InternRobotics/MotionMillion/tree/main/motion_272rpr
- Project reference: https://github.com/VankouF/MotionMillion-Codes
- Expected input: SMPL-family `.npz` files generated from the downloaded
  272-dimensional `.npy` motions
- HoloSMPL source key: `motionmillion`

Convert the downloaded motions first:

```bash
python holomotion/src/data_curation/smplify/smplify_motionmillion.py \
  --src_folder /path/to/motion_272rpr \
  --tgt_folder /path/to/MotionMillion/smplx
```

Then run HoloSMPL:

```bash
python -m holosmpl convert \
  --source motionmillion \
  --input-root /path/to/MotionMillion/smplx \
  --output-root /path/to/out
```

The preparation script reconstructs root motion, converts 6D rotations to
axis-angle, and converts Y-up motion to canonical Z-up.
