# Boxing sample data

English | [简体中文](README.zh-CN.md)

Prepare RoboStriker and InterHuman motions for the boxing trainer and HoloMotion V2.
Run commands from the HoloMotion repository root.

**Samples:** The data and preparation workflow are examples for learning the
process or using as a reference for your own development.

| Source | Preparation route |
| --- | --- |
| RoboStriker | Download released G1 robot NPZ files, then run `prepare` |
| InterHuman | Download paired human SMPL motions, run `retarget`, then run `prepare` |

Use either source or combine both. RoboStriker already contains robot motions;
it does not require SMPL retargeting.

## Environment and resources

Use the [HoloMotion training environment](../../train.env) and
[competition assets](../evaluation/README.md#source-resources).
InterHuman retargeting requires a CUDA GPU and a licensed neutral SMPL body model.
Obtain that model through the [official SMPL download](https://smpl.is.tue.mpg.de/).
It is not included in this repository or the motion dataset.

```bash
source train.env
"$Train_CONDA_PREFIX/bin/python" -m pip install -r holoboxing/data/requirements.txt
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data --help
```

Keep downloaded and generated files in the ignored data directories.
Use the same competition assets throughout data preparation and training.

## Download and convert RoboStriker

Download the G1 motion files released by
[RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview).
The bundled [source profile](profiles/robostriker.json) pins the upstream revision,
download identifiers, file sizes, and SHA256 hashes.

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data download robostriker \
  --output holoboxing/data/raw/RoboStriker --workers 8

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --output holoboxing/data/processed/RoboStriker \
  --assets holoboxing/assets --workers 2

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data verify \
  --output holoboxing/data/processed/RoboStriker --assets holoboxing/assets
```

`prepare` reads the downloaded G1 states directly. It orders joints by name,
resamples to 50 Hz, and derives training records using the competition robot's kinematics.
It does not run human-to-robot retargeting again.
For a small download check, add `--limit-groups 1`; each group keeps an original
motion and its mirrored version together.
`download.json` records source identities and verified file hashes.
Rerunning verifies cached files and resumes interrupted transfers; `--repair`
replaces corrupted cached files.

## Download InterHuman samples

The example uses the bundled [sample file list](profiles/interhuman_training_files.json):
238 clips, comprising 220 official training clips and 18 validation clips.
No official test clips are included. Download these samples from the public source:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data download interhuman \
  --output holoboxing/data/raw/InterHuman --workers 8
```

The default command follows this fixed list without searching annotations.
For a small check, add `--limit-clips 2`. To request different source clips,
repeat `--clip-id ID`; this replaces the bundled list.
`download.json` records the requested IDs, downloaded file hashes, and the
bundled list's hash when used. Existing files are verified; interrupted downloads
resume when rerun. Use `--repair` to replace corrupted cached files.

[InterHuman](https://github.com/tr3e/InterGen#interhuman-dataset) permits
noncommercial use and prohibits dataset redistribution. Download original files
from the official source and retain `LICENSE.md`. Do not redistribute the licensed
SMPL model with your code or data.

## Retarget InterHuman motions

Convert the downloaded paired SMPL motions into G1 motions:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data retarget \
  --source holoboxing/data/raw/InterHuman \
  --output holoboxing/data/converted/InterHuman \
  --smpl-model /path/to/licensed/neutral-smpl.pkl \
  --assets holoboxing/assets --device cuda:0 --iterations 4
```

The source directory must contain the sample download receipt `download.json`.
Retargeting processes only clips recorded in that receipt and preserves official
train, validation, and test assignments. This example uses four HoloRetarget
iterations without changing HoloRetarget's global default.

`motions.json` records source and model hashes, asset identity, and conversion
settings. Outputs contain named 29-joint G1 states at 50 Hz.

## Convert InterHuman to training inputs

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data prepare \
  --source holoboxing/data/converted/InterHuman \
  --output holoboxing/data/processed/InterHuman \
  --assets holoboxing/assets --workers 2

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data verify \
  --output holoboxing/data/processed/InterHuman --assets holoboxing/assets
```

## Combine both sources

To prepare a combined dataset, repeat `--source` in one command:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --output holoboxing/data/processed/boxing \
  --assets holoboxing/assets --workers 2

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data verify \
  --output holoboxing/data/processed/boxing --assets holoboxing/assets
```

Use the original RoboStriker download and retargeted InterHuman directories as inputs,
rather than previously generated `prepared/` directories.
RoboStriker original/mirror pairs share one split. Without official assignments,
the default split uses seed 42 and a validation fraction of 0.1 over source groups.
InterHuman keeps its official assignments; official test clips are excluded.

## Outputs and checks

All `prepare` commands above produce the same output layout.
Use a new output directory outside the source tree.
Outputs containing `FAILED.json` are incomplete.

| Output | Use |
| --- | --- |
| `prepared/` | Current boxing trainer: observations, future pose targets, and motion-prior records |
| `robot_h5/train/` | HoloMotion V2 training references |
| `robot_h5/validation/` | HoloMotion V2 validation references |
| `sources.json`, `COMPLETE.json` | Source lineage, split assignments, and verification results |

Pass the selected dataset's `prepared/` directory to the
[sample training workflow](../training/README.md#environment-and-commands).
RoboStriker contains single-robot references; its boxing observations use a synthetic
standing opponent. InterHuman retains both recorded participants.
Data conversion and reader checks alone do not establish physical tracking or
boxing performance.

## Data and method sources

| Source | What is reused | Scope of this sample |
| --- | --- | --- |
| Liang et al., [InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions](https://arxiv.org/abs/2304.05684), 2024 | The associated InterHuman dataset supplies paired human motion | We retarget selected recorded clips to G1; we do not use InterGen's text-conditioned diffusion model |
| Loper et al., [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/), 2015 | The licensed body model supplies human shape and skeleton information | SMPL represents the source humans; the output uses the competition G1 robot and its joint definitions |
| [HoloRetarget](../../docs/motion_retargeting.md) | The existing human-to-robot retargeting implementation | This adapter handles paired motion, shared horizontal scaling, and the competition robot assets |
| [RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview) | Released G1 motion NPZ files, including mirrored motions | We convert robot states into competition training records and HoloMotion V2 references; no additional SMPL retargeting is needed |

HoloRetarget belongs to the HoloMotion project; its implementation is documented
in the linked guide. The project's research reference is Chen et al.,
[HoloMotion-1 Technical Report](https://arxiv.org/abs/2605.15336), 2026.
That report describes V1; the V2 controller interface follows the current repository documentation.
