# BonesSeedSMPL

- Source type: public SMPL motion release
- Download: https://huggingface.co/nvidia/GEAR-SONIC/tree/main/bones_seed_smpl
- Expected input: extracted `data/smpl_filtered/**/*.pkl` files containing
  `pose_aa`, `transl`, `smpl_joints`, and `fps`
- HoloSMPL source key: `bones_seed_smpl`

The split archive can be downloaded and extracted with the
[GR00T-WholeBodyControl downloader](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/download_from_hf.py):

```bash
python download_from_hf.py --training --output-dir /path/to/GEAR-SONIC
```

Example:

```bash
python -m holosmpl convert \
  --source bones_seed_smpl \
  --input-root /path/to/GEAR-SONIC/data/smpl_filtered \
  --output-root /path/to/out
```

The converter uses `transl` as root translation and converts the source
pose/translation from Y-up to canonical Z-up. `smpl_joints` is used only for
coordinate-axis auditing.
