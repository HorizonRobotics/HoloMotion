# Camera / Video Reconstruction

- Source type: camera/video reconstruction
- Supported method: `gvhmr`
- Video-to-SMPL stage: `holomotion/scripts/data_curation/video_to_smpl_gvhmr.sh`
- Expected input to HoloSMPL: GVHMR-generated SMPL `.npz` files

The complete production flow is video -> GVHMR SMPL NPZ -> HoloSMPL. The
GVHMR inference step uses a separate optional environment; HoloSMPL consumes the
generated SMPL NPZ files and writes the project-standard 50Hz human-side
representation.

```text
RGB video -> GVHMR SMPL NPZ -> HoloSMPL -> HoloRetarget robot HDF5
```

Dataset conversion for AMASS, MotionMillion, and other supported motion sources
does not require GVHMR. Do not install GVHMR unless you need monocular
video-to-SMPL reconstruction.

## Set Up GVHMR

Create the `gvhmr` Conda environment by following the
[GVHMR installation guide](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md).
Install `hmr4d` when it is not already present, and install `pywebview` for the
optional SMPL viewer:

```bash
conda activate gvhmr
python -m pip install hmr4d pywebview
```

Place the required SMPL and SMPL-X models under the GVHMR checkpoint tree:

```text
thirdparties/GVHMR/inputs/checkpoints/
├── body_models/smpl/
│   └── SMPL_{GENDER}.pkl
└── body_models/smplx/
    └── SMPLX_{GENDER}.npz
```

Rename downloaded SMPL files from `basicmodel_{GENDER}_lbs_*.pkl` to
`SMPL_{GENDER}.pkl` when needed.

## Convert Videos

Use 30 FPS input videos to preserve motion timing. By default, place videos
under `data/video_data`, then run from the repository root:

```bash
bash holomotion/scripts/data_curation/video_to_smpl_gvhmr.sh
```

The script writes GVHMR results under `data/gvhmr_converted/gvhmr_result` and
collects the generated SMPL files under
`data/gvhmr_converted/collected_smpl`.

Override paths or the Conda environment without editing the script:

```bash
GVHMR_CONDA_PREFIX=/path/to/gvhmr \
VIDEO_FOLDER_ROOT=/path/to/videos \
NPZ_DATA_ROOT=/path/to/gvhmr_results \
SMPL_OUTPUT_ROOT=/path/to/collected_smpl \
  bash holomotion/scripts/data_curation/video_to_smpl_gvhmr.sh
```

## Inspect and Continue

Inspect generated SMPL sequences before conversion:

```bash
bash holomotion/scripts/data_curation/visualize_smpl_npz.sh
```

The viewer uses the `gvhmr` Conda environment by default. Override its location
when needed:

```bash
GVHMR_CONDA_PREFIX=/path/to/gvhmr \
  bash holomotion/scripts/data_curation/visualize_smpl_npz.sh
```

Then feed the SMPL NPZ files into HoloSMPL:

```bash
python -m holosmpl convert \
  --source gvhmr \
  --input-root data/gvhmr_converted/collected_smpl \
  --output-root /path/to/holosmpl_output \
  --overwrite
```

Finally, use [HoloRetarget](../../../docs/motion_retargeting.md) to produce
robot HDF5 training data.
