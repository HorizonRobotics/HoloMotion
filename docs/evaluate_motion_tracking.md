# Evaluate Motion Tracking

HoloMotion v2.0 provides MuJoCo simulation and quantitative evaluation on
user-provided motion data. The current release does not include the fixed
evaluation dataset used for our benchmark results. The v2.0 evaluation dataset
may be released in the future. The evaluator is not tied to a particular
dataset, clip count, or set of evaluation groups.

Run the commands below from the HoloMotion repository root after setting up
the [training environment](environment_setup.md). Input formats are described
in the [motion file specification](holomotion_motion_file_spec.md).

## 1. Prepare the model and motion data

Download the complete
[HoloMotion v2.0.0 motion-tracking bundle](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_motion_tracking_model_v2.0.0)
into `checkpoints/HoloMotion_motion_tracking_model_v2.0.0/`, preserving its
layout: `config.yaml`, `link_command_modes.json`, and
`exported/model_10500.onnx`. The simulation examples below use this path.

The published bundle currently contains inference artifacts only. It does not
include the actor weights required by the HDF5 checkpoint-comparison step or
the complete training checkpoint required for fine-tuning. For quantitative
evaluation, provide the matching actor weights in the layout below.

- Use an exported ONNX policy and the matching saved training `config.yaml`
  in its run directory. Do not mix model weights, DOF order, or observation
  schemas from different runs.
- Install the G1 robot assets and use the model's saved FK asset and DOF order.
- For interactive simulation, provide a retargeted motion NPZ.
- For quantitative evaluation, provide root/DOF HDF5 shards, their manifest,
  and an explicit list of clips. Keep the matching actor checkpoint alongside
  the ONNX for the one-time checkpoint comparison.

The quantitative path supports the integrated v2.0 link-observation actor.
It rejects unsupported saved architectures instead of approximating their
observations. An ONNX file alone is not a complete quantitative-evaluation
bundle.

Keep the bundle in the layout used by the evaluator:

```text
run/
├── config.yaml
├── exported/model_500.onnx
└── model_500/actor/model.safetensors
```

The actor file may instead be `pytorch_model.bin`. Quantitative evaluation
locates it under `run/<ONNX filename without extension>/actor/`, so preserve
the matching names when moving an export. Interactive NPZ simulation needs
the ONNX and configuration; HDF5 evaluation additionally needs the actor
weights for its checkpoint comparison. The training-state `.pt` file alone
does not contain these actor weights.

## 2. Sim2sim: inspect a reference motion

Use the existing simulation script with your own paths:

```bash
ONNX_PATH=checkpoints/HoloMotion_motion_tracking_model_v2.0.0/exported/model_10500.onnx \
MOTION_NPZ_PATH=/path/to/retargeted_motion.npz \
  bash holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh
```

The script accepts trailing Hydra overrides. For example, select pelvis-only
control for a v2.0 policy:

```bash
ONNX_PATH=checkpoints/HoloMotion_motion_tracking_model_v2.0.0/exported/model_10500.onnx \
MOTION_NPZ_PATH=/path/to/retargeted_motion.npz \
  bash holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh \
  link_eval.mask_mode=custom 'link_eval.active_link_names=[pelvis]'
```

`HEADLESS=true` disables the viewer and defaults to video recording.
`RECORD_VIDEO=false` disables recording; `ROBOT_XML_PATH` overrides the scene.
`USE_GPU=false` selects CPU inference. Simulation is useful for inspecting
motion quality, but watching a clip is not a complete dataset score.
See [sim2sim verification](mujoco_sim2sim.md).

The NPZ must contain retargeted **robot** reference data as defined in the
[file specification](holomotion_motion_file_spec.md#npz--keys-and-values),
including joint positions/velocities and body poses. A human-side HoloSMPL
NPZ or the compact root/DOF arrays from a training HDF5 shard cannot be passed
directly as `MOTION_NPZ_PATH`. This workstation simulation entry is separate
from offline replay in the robot deployment image.

## 3. Quantitative evaluation on your HDF5 dataset

Use the same MuJoCo entry with the compact root/DOF backend:

```bash
source train.env
"$Train_CONDA_PREFIX/bin/python" -m holomotion.src.evaluation.eval_mujoco_sim2sim \
  --config-name=evaluation/eval_mujoco_sim2sim \
  ckpt_onnx_path=/path/to/run/exported/model_500.onnx \
  robot_xml_path=assets/robots/unitree/G1/29dof/scene_29dof.xml \
  evaluation_data.backend=hdf5_v2 \
  evaluation_data.manifest_path=/path/to/my_dataset/manifest.json \
  evaluation_data.selection_path=/path/to/my_dataset/selection.json \
  evaluation_run.output_dir=/path/to/new-evaluation \
  evaluation_run.workers=1 evaluation_run.gpus=1 headless=true
```

`hdf5_v2` is the existing compact root/DOF format used by training, not a
requirement to use a specific benchmark dataset. The older full-state
`hdf5` format is not supported by this adapter. Each selected clip is loaded
as one complete window; FK and normalization stay in memory, with no
per-clip NPZ input copies.

Specify either `evaluation_data.selection_path` or
`evaluation_data.clip_names='[my_clip_a,my_clip_b]'`, never both.
The evaluator requires an explicit, nonempty clip list; it does not scan the
directory or apply training-time source sampling and random masks.
Use data independent of training, and keep the selected clips and control
points unchanged when comparing models instead of resampling for each model.

A `selection.json` uses the loader's existing
`val_handpicked_motion_names` field. Here it names evaluation clips, not
necessarily a training-validation split:

```json
{
  "val_handpicked_motion_names": ["my_clip_a", "my_clip_b"]
}
```

Names must be unique and present in the manifest. An optional
`manifest_sha256` binds the selection to the SHA256 of the manifest's raw
file bytes; `require_manifest_fingerprint=true` makes that field mandatory.
If `train_handpicked_motion_names` is also provided, it must not overlap
the evaluation list. Training does not require this selection file.
`expected_clips` can assert your intended count; no fixed count is assumed.
Before running, check shard availability, clip frame bounds, FPS and DOF
order. Every clip must be longer than the model's future-reference horizon.

`evaluation_data.reference_horizon` must equal the model's maximum saved
future-frame offset. FK defaults to the saved robot asset; use
`fk_robot_file_path` only for an intentional, compatible override.
For parallel evaluation, set `evaluation_run.workers` and
`evaluation_run.gpus` (currently 1–4 GPUs). GPU evaluation rejects silent CPU
fallback. For CPU validation, use `use_gpu=false` and
`evaluation_run.gpus=1`.

This path evaluates one explicit ONNX checkpoint at a time, with
`metric_calculation=per_clip` and `max_policy_steps=0`. It does not record
video, replay actions, add action delay, dump ONNX I/O, or generate an HTML
report. NPZ-only `dof_mode`, `ray_*`, `dump_npzs`,
`calc_per_clip_metrics`, and `metrics_threadpool_max_workers` options are
ignored with a startup warning. Use the simulation path for visualization.

## 4. Select control points

Both simulation and HDF5 evaluation accept `link_eval`. The default
`mask_mode: all14` commands all fourteen links in the saved schema.
For a subset, use explicit unique names from that schema:

```yaml
link_eval:
  mask_mode: custom
  active_link_names: [pelvis, left_wrist_yaw_link, right_wrist_yaw_link]
```

A one-point input uses `[pelvis]`; five-point control adds both ankle-roll
links to the three-point example. The observation mask and quantitative
metric point selection share the training mask resolver. SR always measures
root height, even if pelvis is not among the commanded points.

To evaluate several subsets on the same clips, set
`evaluation_run.groups` to a list of `{name: ..., link_eval: ...}` mappings.
Otherwise the single group uses `evaluation_run.phase` and the top-level
`link_eval`. Names and the number of groups are user-defined.

## 5. Metrics, aggregation and outputs

- SR: a clip succeeds when its maximum same-frame root-height error is at
  most `failure_pos_err_thresh_m` (default 0.25 m).
  SR = successful clips / evaluated clips.
- Global MKPE: world-coordinate position error over active points and
  scored frames, averaged equally over clips.
- Local MKPE: subtract each pose's own pelvis translation, retaining
  orientation error. Pelvis-only local error is structurally zero.
  This is not the root-rotation-aligned `mpjpe_l` reported by the NPZ metrics.
- MKPE values are in meters. Complete clips that fail SR still contribute
  position errors. Incomplete or invalid rollouts do not produce a successful
  dataset summary.

The runner first checks ONNX against the matching checkpoint for eight steps
in an isolated FP32 session. This is one check per run, not per inference.
Rollout workers then run without another checkpoint comparison; GPU rollout
uses configurable TF32, enabled by default.

Each clip starts from its reference pose and velocity with fresh observation,
previous-action and recurrent state. Metrics are collected before control for
every usable frame through `clip_length - reference_horizon - 1`.
An SR failure does not terminate rollout.

`aggregate_evaluation_results.py` is a generic result aggregator, not a
benchmark webpage generator. The runner automatically calls
`aggregate_shards(...)` even with one worker: it checks clip/frame/control
coverage, merges worker results, and computes dataset metrics. Users do not
need a separate aggregation command. Missing, duplicate, truncated,
nonfinite, or mismatched-control results fail validation.

The result directory contains:

```text
<output>/
├── resolved_config.yaml
├── identity.json
├── checkpoint_parity.json
├── summary.json
├── done.json
└── <group>/
    ├── results/worker*/per_clip.json
    └── aggregate/{coverage,summary,per_clip}.json
```

`summary.json` contains all requested group summaries.
Within a group, `root_height_SR` holds success statistics and
`equal_clip_common_metrics_completed_only` holds position-error aggregates.
`done.json` is written only after all groups pass. Errors write
`failed.json`; partial outputs are not a complete score. Use a fresh output
directory that does not overlap the input data or checkpoint directories.

## 6. Existing offline evaluation tools

The IsaacLab offline evaluator and NPZ post-processing tools remain available
for compatible checkpoints and user-provided motions:

```bash
CKPT_PATH=/path/to/run/model_500.pt \
EVAL_H5_DATASET_PATH="['/path/to/my_dataset']" \
  bash holomotion/scripts/evaluation/eval_motion_tracking.sh
```

For an existing folder of NPZ rollout results:

```bash
source train.env
"$Train_CONDA_PREFIX/bin/python" -m holomotion.src.evaluation.metrics \
  --npz_dir=/path/to/npz_results \
  --dataset_suffix=my_dataset \
  --failure_pos_err_thresh_m=0.25 --metric_calculation=per_clip --dof_mode=29
```

NPZ whole-body metrics and HDF5 active-point metrics have different point and
local-coordinate definitions. Do not treat their numbers as interchangeable.
Use the HDF5 route above for active-point quantitative evaluation.

Dataset release, platform submission and multi-model benchmark reports are
separate from these public runtime interfaces. CPU regression tests do not
establish production GPU performance or real-robot behavior.
