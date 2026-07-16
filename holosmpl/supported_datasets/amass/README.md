# AMASS

- Source type: public dataset
- Download: https://amass.is.tue.mpg.de/download.php — select `SMPL-X G`; skip
  `BMLhandball`, which has no `SMPL-X G` release
- Expected input: extracted AMASS motion `.npz` tree
- HoloSMPL source key: `amass`

Example:

```bash
python -m holosmpl convert \
  --source amass \
  --input-root /path/to/AMASS/raw \
  --output-root /path/to/out
```

The converter reads SMPL-family root/body pose, translation, shape, and FPS
fields. Hand and face fields are not used.
