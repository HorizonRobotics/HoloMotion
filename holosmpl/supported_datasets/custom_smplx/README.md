# CustomSMPLX

- Source type: user-defined SMPL-X motion
- Download: private data
- Expected input: packed `.pkl` sequences containing:
  - `poses [T,66]`: root orientation and 21 body-joint axis-angle pose
  - `trans [T,3]`: root translation
  - `joints [T,22,3]`: body joint positions
  - `betas [B]`: body shape coefficients
- HoloSMPL source key: `custom_smplx`

Example:

```bash
python -m holosmpl convert \
  --source custom_smplx \
  --input-root /path/to/custom_smplx \
  --output-root /path/to/out
```

Private SMPL-X data can be integrated in either of the following ways:

- **Implement a source converter:** use the Expected input contract above and
  the [custom_smplx converter](../../converters/smpl_family/custom_smplx.py)
  as references. Implement a converter for your own packing layout, frame
  rate, and coordinate system, then use HoloSMPL to produce the standard
  output.
- **Construct HoloSMPL directly:** if your preprocessing already provides all
  required HoloSMPL fields, write the NPZ/H5 data directly according to the
  [HoloSMPL schema](../../README.md#data-format).
