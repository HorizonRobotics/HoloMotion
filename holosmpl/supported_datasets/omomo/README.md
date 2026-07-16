# OMOMO

- Source type: public dataset
- Download: https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view?usp=sharing
- Project reference: https://github.com/lijiaman/omomo_release
- Expected input: SMPL-family `.npz` files generated from the downloaded data
  by `holomotion/src/data_curation/smplify/smplify_omomo.py`
- HoloSMPL source key: `omomo`

Set `data_root_folder` and `target_folder` at the bottom of the preparation
script, then run it before HoloSMPL.

Example:

```bash
python -m holosmpl convert \
  --source omomo \
  --input-root /path/to/OMOMO_npz \
  --output-root /path/to/out
```

The converter reads `poses`, `trans`, `betas`, `gender`, and FPS fields. It
uses the root and body portion of each pose vector; hand and face pose
dimensions are not included in the HoloSMPL body output.
