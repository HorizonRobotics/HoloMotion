# HoloBoxing

English | [简体中文](README.zh-CN.md)

**Follow the competition's evaluation interface and rules. Data preparation and
training examples are optional; you can develop your policy independently.**

| Content | What it means for participants |
| --- | --- |
| Required contract | Submit an ONNX model that satisfies the policy interface, file limits, and official hardware/latency requirements; matches follow the competition scoring rules |
| Local evaluation | Test with the same evaluation image used in the cloud: validate a model, run matches, and watch replays; recommended before submission |
| Optional references | `data/` and `training/` demonstrate a development workflow; you may use your own data, network architecture, and training method |

ONNX is the portable neural-network format used for submission.
An existing compatible ONNX policy can be evaluated directly, without running
the data or training examples.

## Test your policy

Local and cloud evaluation use the same evaluation image. Use the official
release's exact image digest to keep evaluation code, dependencies, scoring
rules, frozen HoloMotion controller, and resources aligned.

Validation and matches require Linux, Docker, an NVIDIA GPU, and NVIDIA
Container Toolkit. Playback requires Docker and a browser.
The image bundles the tracker, robot resources, arena, and player; no separate
download or host Python environment is needed.
The v9 image includes the gloved-robot and collision-precision fixes.
Its published tag is `horizonrobotics/holomotion:holoboxing-eval-v9-amd64`.
The commands below pin its verified repository digest.
Pull the published image before running these examples:

```bash
export HOLOBOXING_IMAGE="horizonrobotics/holomotion@sha256:5c54be599351bafe2392c8fdae2647ba6c107a760228bffa8b2b33dd85e2a67a"
docker pull --platform linux/amd64 "$HOLOBOXING_IMAGE"
```

See the [setup guide](evaluation/README.md#run) for digest pinning, archive loading, and host requirements.
Source development uses the [published resource revision](evaluation/README.md#source-resources) pinned by this checkout.

`my_policy.onnx` is your model; `opponent_policy.onnx` is a model to test against.
Replace both example paths with your actual files.
The first model controls the red robot and the second controls blue.
Run from the HoloMotion repository root:

```bash
./holoboxing/run.sh --validate models/my_policy.onnx
./holoboxing/run.sh --match models/my_policy.onnx models/opponent_policy.onnx
./holoboxing/run.sh --view results/match/replay.hdf5
```

Validation writes `results/validation.json`. Matches write scores and events to
`results/match/match.json`, alongside `replay.hdf5` for the browser player.
Use `--output` for another result path; existing results are never overwritten.
Local commands do not submit your model or update the competition leaderboard.

## What to submit

Submit one self-contained ONNX model for your high-level boxing policy.
The evaluator supplies the frozen HoloMotion V2 low-level tracker.

| Requirement | Value |
| --- | --- |
| Model file | One `.onnx` file, strictly smaller than 50,000,000 bytes; no external tensor files |
| Input | `obs`, float32, shape `[1,1097]`: self-state history, opponent positions, remaining time, and score difference |
| Output | `target_pose`, float32, shape `[1,225]`: five future steps × five body points × nine pose values |
| Control rate | 50 Hz; each action covers targets 20–100 ms ahead |
| Initial-round latency gate | Complete policy-and-tracker chain mean ≤ 5 ms on an RTX 5090 |

See the [policy interface](evaluation/README.md#policy-interface) for observation
indices, point order, coordinate frames, rotation representation, and export restrictions.
The [validation guide](evaluation/README.md#validation-and-errors) defines timing
conditions and the official hardware check.

## How a match is scored

A round lasts up to 60 simulation seconds. A qualifying hand contact with the
opponent's head or torso earns one point. A confirmed knockout earns 1,000 points
and ends the round. Otherwise, the higher score wins at timeout; equal scores draw.

See [match rules](evaluation/README.md#match-rules) for contact thresholds,
scoring cooldowns, fall criteria, and simultaneous knockouts.

## Why use HoloMotion?

The required policy interface separates boxing decisions from whole-body execution:

- **Specify five body targets.** Your policy commands the pelvis, both wrists,
  and both ankles. HoloMotion converts these pose targets into coordinated
  actions for the robot's 29 joints. You can express punches, defense, and
  movement through body goals without directly specifying every joint action.
- **Reuse motion pretraining.** The pretrained tracker supplies whole-body motion
  tracking capabilities. It stays frozen, so you can focus on high-level boxing
  policy development without training a motion tracker from scratch.

Five-point control simplifies how you express the task. Each output still
contains five future sets of point poses, totaling 225 values.
See the [HoloMotion V2 overview](../README.md#holomotion-v2-flexible-sparse-to-whole-body-command-following-with-cross-modal-extensibility)
and [pinned controller bundle](evaluation/frozen-model.json) for the underlying
controller and model version. Boxing performance depends on your policy and its targets.

## Optional data and training examples

To make development easier to start, we provide sample data preparation and
training workflows. They demonstrate data preparation, supervised initialization,
reinforcement learning, ONNX export, and evaluation.

These are development references, not required competition steps. You may use
your own data and training implementation, provided the exported policy meets
the evaluation contract. The samples do not target state-of-the-art (SOTA) boxing performance.
Start with the [sample data guide](data/README.md) and
[sample training guide](training/README.md) if you want to follow this reference workflow.
Source workflows share the [resource setup](evaluation/README.md#source-resources).

### Sample methods and sources

The reference training workflow adapts the sandbag-training structure and parts
of the [RoboStriker](https://arxiv.org/abs/2608.16195) implementation.
Both use latent representations. This sample decodes an internal 32D latent
into 225D five-point targets for frozen HoloMotion V2. PPO samples those targets;
RoboStriker's strategic actions are latent variables. This sample architecture
is not a requirement for participant policies.

See the [training references](training/README.md#methods-and-sources),
[data references](data/README.md#data-and-method-sources),
[resource attribution](assets/smpl-ring-v2/SOURCE.md), and
[rotation convention](evaluation/README.md#targets) for the reused components and adaptations.

## Directory guide

| Directory | Purpose |
| --- | --- |
| [evaluation/](evaluation/README.md) | Common evaluation interface, match rules, model validation, matches, and replay |
| [assets/](assets/README.md) | Robot, arena, and replay resources; already bundled for Docker evaluation |
| [data/](data/README.md) | Optional sample data preparation workflow |
| [training/](training/README.md) | Optional sample policy training and ONNX export workflow |
