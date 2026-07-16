# LaFAN1

- Source type: public dataset
- Download: https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip
- Expected input: extracted LaFAN1 `.bvh` files
- HoloSMPL source key: `lafan1`

Example:

```bash
python -m holosmpl convert \
  --source lafan1 \
  --input-root /path/to/LaFAN1 \
  --output-root /path/to/out
```

The converter uses an approximate BVH-to-SMPL-X-like joint mapping adapted
from [lafan_to_smplx](https://github.com/jaraujo98/lafan_to_smplx).
