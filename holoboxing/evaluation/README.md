# HoloBoxing evaluation

English | [简体中文](README.zh-CN.md)

Validate a high-level boxing policy, run a two-robot match with the frozen
HoloMotion V2 tracker, and replay the recorded result. Playback reads saved
states and events without running policy inference or stepping physics.

| Component | Main functions |
| --- | --- |
| `cli.py`, `validation.py`, `qualification.py`, `benchmark.py` | Command-line entry, model interface checks, hardware qualification, and complete-chain inference timing |
| `match.py`, `runtime/` | Upper-policy and frozen-tracker execution, GPU physics, hit and fall rules, scoring, and match diagnostics |
| `export_hdf5.py`, `replay_schema/`, `viewer/`, `viewer_server.py` | Store replay data, serve the browser player, and export video with recorded event sounds |
| `resources.py`, `models.py`, `frozen-model.json` | Verify resource inventories and the pinned lower-controller bundle |
| `build.sh`, `scripts/`, `docker/` | Prepare an allowlisted build context and build the bundled evaluation image |

Use `./holoboxing/run.sh` from the HoloMotion repository root for Docker-based
validation, matches, and playback. The Python entry is `python -m holoboxing`.
Training commands live in [`training/`](../training/README.md).
The data and training guides provide samples for learning the workflow or using
as a reference.

## Guide

- [Run](#run).
- [Match rules](#match-rules).
- [Policy interface](#policy-interface).
- [Validation and errors](#validation-and-errors).
- [Build the image](#building-the-image).

For training or source evaluation, follow [source resources](#source-resources).

**We strongly recommend the official evaluation image from Docker Hub, pinned
to the exact image digest used by cloud evaluation.** This keeps evaluation
code, dependencies, the frozen HoloMotion tracker, and resources identical to the cloud
runtime. A matching tag alone does not guarantee identical contents.

The image includes the **HoloMotion V2 checkpoint, robot URDFs, meshes, arena
textures, and browser player**. These resources need no separate download.
You supply only your high-level boxing policy as an ONNX model, a portable neural-network file.

## Run

Use Linux, Docker, an NVIDIA driver, and NVIDIA Container Toolkit for validation
and matches. The Toolkit gives containers GPU access. Replay playback needs
only Docker and a browser. Host Python and npm are unnecessary for image users.

The host workflow was tested on **Ubuntu 22.04.4 LTS with Docker 27.1.2**.
The evaluation image runs Debian 12 (Bookworm). Official evaluation requires
an **RTX 5090**. Other host operating systems have not been validated here.

Use the published Linux/amd64 image [holoboxing-eval-v8-amd64](https://hub.docker.com/r/horizonrobotics/holomotion/tags?name=holoboxing-eval-v8-amd64).
Pin its digest with the following commands:

```bash
export HOLOBOXING_IMAGE="horizonrobotics/holomotion:holoboxing-eval-v8-amd64@sha256:d5532acccf8a1e642b452da1d1a09bbbb1e24301ff61a1ffd5b9a2180fc78ec4"
docker pull "$HOLOBOXING_IMAGE"
```

`./holoboxing/run.sh --init` remains a reserved setup entry and performs no installation
or download. Pull the image explicitly as shown above.
Run the following commands from the HoloMotion repository root.

Use your model as `my_policy.onnx` and another model as `opponent_policy.onnx`.
Replace the example paths with actual files.
The first model controls the red robot; the second controls the blue robot.

```bash
./holoboxing/run.sh --validate models/my_policy.onnx
./holoboxing/run.sh --match models/my_policy.onnx models/opponent_policy.onnx
./holoboxing/run.sh --view results/match/replay.hdf5
```

For an image archive, import it with `docker load --input image.tar` and select
its verified tag or image ID. Source builds are intended for development;
they do not establish parity with the cloud's pinned image.

| Task | Output |
| --- | --- |
| `--validate MODEL` | `results/validation.json`: model checks, measured complete-chain latency, hardware qualification |
| `--match YOUR_POLICY OPPONENT_POLICY` | `results/match/match.json`: scores, winner, events, diagnostics; `replay.hdf5`: recorded states |
| `--view REPLAY` | Player at `http://localhost:8080`: play, pause, seek, event markers, MP4 export |

Results are never overwritten. Use `--output` for another result path,
`--gpu` for another host GPU, or `--port` for another local player port.
Run `./holoboxing/run.sh --help` for all options. Models are mounted read-only;
containers run as your current user. `--assets` and `--lower` optionally replace
the bundled resources and tracker; normal use needs neither option.

Playback reads saved states without executing policies or stepping physics.
Recorded `HIT` and `KO` events trigger the selected punching-bag recording and
female K.O. voice. Both recordings use CC0; credits and processing are bundled
in the viewer's `notices/recorded-effects.txt`. Seeking stays silent; the sound
switch controls both playback and MP4 export. Export retains the terminal voice's tail.
Keep the browser tab visible while exporting MP4; stop the server with Ctrl+C.
MP4 encoding requires HTTPS or localhost in a compatible Chrome / Edge browser.
For a remote server, forward its player port through SSH to localhost, or run
`python -m holoboxing view replay.hdf5 --host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem`
with a certificate trusted by the browser. Keep the private key outside the repository.


## Match rules

One round lasts at most **60 simulation seconds**. Control runs at 50 Hz
(one policy target every 20 ms), with physics at 200 Hz (one step every 5 ms).
Simulation duration differs from the command's wall-clock runtime.

| Decision | Definition |
| --- | --- |
| Hit | A hand contacts the opponent's head or torso, accumulating at least **10 N·s** normal impulse within **100 ms**; award **1 point** |
| Repeat hit | Each hand has a **100 ms** scoring cooldown. Scoring consumes its accumulated impulse; contact separation is unnecessary |
| Fall | The pelvis or torso body origin is strictly below **0.30 m** above the arena floor. Tilt alone does not count |
| Knockout (KO) | A continuous fall lasts **3 s**; award the opponent **1,000 points** and end the round |
| Winner | A single confirmed KO wins. Simultaneous confirmed falls draw. At timeout, the higher score wins; equal scores draw |

Normal impulse integrates the contact force perpendicular to the surface over
time, in newton-seconds (N·s). Friction, peak force, and closing speed do not
replace that threshold. Inspect recorded events and contact diagnostics to explain a score.
The result records the scoring version and numeric configuration.
Fall ordering uses the 50 Hz observation timestamps. Identical timestamps are
simultaneous; stored event array order never breaks a tie.


## Policy interface

Your **high-level boxing policy** predicts future poses for five body points. The bundled,
**frozen HoloMotion low-level tracker** converts these targets into joint actions.
Submit only the high-level boxing policy; its file must be self-contained and strictly
smaller than **50,000,000 bytes**. External tensor files are unsupported.

A tensor is a numbered array. The runtime uses batch size 1 and float32 values
without automatic normalization. A symbolic batch dimension may support size 1.

| Direction | Tensor name | Type | Shape |
| --- | --- | --- | --- |
| Input | `obs` | float32 | `[1,1097]` |
| Output | `target_pose` | float32 | `[1,225]` |

Optional ONNX metadata `policy_contract`, when present, must equal
`holomotion_v2_boxing_obs1097_target_pose225_v1`.

Export in inference mode. Random ONNX operators and training-mode Dropout are
unsupported, including inside branches and local functions. Wide TopK selections
require a known last-axis width and constant `K` between 1 and 128.
Unsupported selections fail validation.

### Coordinate frames

The **controlled robot** runs this policy; the **opponent** is the other robot.
All coordinates below use the controlled robot as their reference:

| Frame | Origin | Axis orientation |
| --- | --- | --- |
| Controlled robot's pelvis frame | Its current measured pelvis position | Follows its pelvis's full 3D rotation: roll, pitch, and yaw |
| Controlled robot's heading frame | Its current measured pelvis position | Follows only its pelvis yaw; Z stays vertical in world coordinates |

### Observations

Slices use zero-based, half-open indexing. Flattening varies the last axis fastest.

| Slice in `obs` | Meaning | Shape before flattening |
| --- | --- | --- |
| `[0:1080]` | Ten measured self states, oldest to current, spaced 20 ms apart | `[10,108]` |
| `[1080:1095]` | Opponent point positions expressed in the controlled robot's current pelvis frame | `[5,3]` |
| `[1095:1096]` | Remaining round time, seconds | `[1]` |
| `[1096:1097]` | Controlled robot's score minus opponent's score | `[1]` |

Each 108-value self state contains:

| Slice within one state | Meaning | Shape |
| --- | --- | --- |
| `[0:3]` | Pelvis angular velocity in its sampling-time body frame, radians/second | `[3]` |
| `[3:48]` | Five measured point poses in the controlled robot's current heading frame | `[5,9]` |
| `[48:63]` | Point linear velocities in the controlled robot's current heading frame, metres/second | `[5,3]` |
| `[63:108]` | The preceding sample's point poses in the controlled robot's current heading frame | `[5,9]` |

Historical poses use the controlled robot's **current** heading frame, including its current origin.
Linear velocities use its current heading axes without subtracting pelvis velocity. Angular velocity keeps its original
sampling-time body frame. The current state occupies `obs[:,972:1080]`.
After reset, every history slot repeats the first measured state; its preceding
pose equals its current pose. Observation precedes this step's action.

### Targets

Reshape `target_pose` to `[1,5,5,9]`: batch, future time, body point, pose component.

| Axis | Order |
| --- | --- |
| Future time | 20, 40, 60, 80, 100 ms after the current observation |
| Body point | Pelvis, left wrist, right wrist, left ankle, right ankle |
| Exact body names | `pelvis`, `left_wrist_yaw_link`, `right_wrist_yaw_link`, `left_ankle_roll_link`, `right_ankle_roll_link` |
| Pose components | `px,py,pz,Xx,Xy,Xz,Zx,Zy,Zz` |

Each point pose combines **3D position + 6D rotation**, giving nine values.
Positions use metres. The rotation follows the continuous 6D representation
introduced by Zhou et al.,
[*On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019](https://arxiv.org/abs/1812.07035).

This interface stores the rotation matrix's **first and third columns**:
`X = (Xx,Xy,Xz)` and `Z = (Zx,Zy,Zz)`. Recover the second column as
`Y = Z × X`, giving `R = [X Y Z]`. The model must output orthogonal unit X/Z
axes; the evaluator checks them with tolerance `1e-3`.

Zhou et al.'s paper motivates the continuous 6D representation.
The X/Z column order above is this interface's convention; do not infer the
packing from another implementation's 6D field order.

Every future target is expressed in the controlled robot's current pelvis frame.

Nonfinite output positions become zero. A nonfinite orientation component
replaces that point's X/Z axes with identity axes; diagnostics count these repairs.
Finite invalid orientations remain errors. Measured states, physics, frozen
features, and recurrent caches are never repaired. The low-level tracker receives
all five future point sets; its packing, validity masks, weights, and state remain fixed.


## Validation and errors

Local validation tests compatibility on your actual GPU. It does not establish
policy skill or official qualification. A fixed-standing policy may legitimately
finish 0:0. Inference and physics have no CPU fallback.

The initial-round hardware gate requires an **RTX 5090** and a complete-chain
mean of at most **5 ms** after 100 warmup steps and 1,000 measured steps.
The timed chain includes high-level observations, inference, decoding, tracker
observations, inference, cache updates, and target conversion. Output repair
and physics are measured separately. Returned recurrent state is reused and
reset according to the frozen HoloMotion tracker's context length.
The observation timing includes the independent forward-kinematics refresh.

The report records actual hardware, software, model hashes, latency percentiles,
and parity checks. `--qualification official` checks the hardware gate explicitly;
a different GPU reports `PLATFORM_WRONG_TARGET_GPU`. Measurements from other GPUs
cannot be converted by a fixed multiplier. Local matches do not update a competition leaderboard.

For the complete control-frame deadline, run the separate budget stress probe:

```bash
python -m holoboxing.evaluation.benchmark my_policy.onnx opponent_policy.onnx \
  --assets /path/to/assets --lower /path/to/lower \
  --output-dir /path/to/new-benchmark --policy-budget-ms 10
```

It first checks each complete upper-policy + HoloMotion v2.0 chain against the
existing 5 ms mean criterion. It then runs a complete match, reserving a combined
10 ms for the two policy chains inside every measured frame. Actual models and
state transitions still execute. The reservation uses wall-clock padding and
counts scheduler overshoot; it does not simulate the GPU load of every possible
10 ms policy. Output sanitation, physics, referee and in-memory replay preparation
are additional frame work. Cold setup, final file encoding and browser playback
are outside this steady-state boundary.

The command fails unless completed-frame P99 is **strictly below 20 ms** after
100 warmup frames. The percentile uses the next observed sample (`higher`),
without interpolating across a missed deadline. Reports preserve model hashes,
actual GPU/driver/load, single-chain measurements and every reservation sample.
Re-run the gate on the target GPU and repeat complete matches for tail evidence.

| Symptom or error | Action |
| --- | --- |
| `Image unavailable` | Import the provided image; check `--image` or `HOLOBOXING_IMAGE` |
| `File does not exist` | Check your working directory and input path |
| `Output already exists` | Choose another `--output` |
| CUDA/GPU unavailable | Check the driver, Container Toolkit, and Docker GPU access |
| `MODEL_TOO_LARGE` | Keep the self-contained file strictly below the size limit |
| `FORMAT_CHECK_FAILED` | Export a valid ONNX file without external tensor storage |
| `CONTRACT_CHECK_FAILED` | Check names, dimensions, types, and optional contract metadata |
| `PLATFORM_WRONG_TARGET_GPU` | Use default local mode for compatibility; use the required GPU for official qualification |
| Robot falls | Check predicted targets and recorded states against the interface and fall rule |


## Source resources

The published boxing resources are in
[HorizonRobotics/HoloMotion_assets](https://huggingface.co/datasets/HorizonRobotics/HoloMotion_assets/tree/main/holoboxing).
HoloMotion already tracks this dataset at `thirdparties/HoloMotion_assets`.
The `holoboxing/assets` symbolic link points to its `holoboxing/` subdirectory.
The submodule revision is pinned by this HoloMotion checkout.
See the [asset guide](../assets/README.md) for file layout and attribution.

With Git LFS installed, run from the HoloMotion repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init thirdparties/HoloMotion_assets
git -C thirdparties/HoloMotion_assets lfs pull --include="holoboxing/**" --exclude=""
```

These commands download the boxing payload without fetching other large model or data files.
The repository's small files and other LFS pointer files are also checked out.
Use `--assets holoboxing/assets` for source evaluation, training, or image builds.
The dataset repository root is not the boxing resource root.
The frozen V2 controller is a separate download.
Docker evaluation images already contain both resources and the controller.

Verify downloaded resources in the HoloMotion training environment:

```bash
source train.env
"$Train_CONDA_PREFIX/bin/python" -m holoboxing assets --assets holoboxing/assets
```


## Building the image

Maintainers need Docker with BuildKit, the inventoried boxing resource tree,
and the frozen bundle identified by [frozen-model.json](frozen-model.json).
Source-only users can fetch that bundle with `python -m holoboxing download --directory /absolute/lower`.
Build from a clean, committed HoloMotion checkout:

```bash
./holoboxing/evaluation/build.sh --assets /absolute/assets --lower /absolute/lower --context /absolute/new-context --image holoboxing-eval:local
```

Use a new context outside the source repository. The script pins public base
image digests and verifies resources and model hashes. `--help` exposes overrides;
`--development` explicitly records a development snapshot. It never publishes an image.
Set `HOLOBOXING_BUILD_PYTHON` to select the build helper's Python interpreter.

The allowlisted context excludes tests and training code. PyTorch is
used only while exporting and checking canonical observation graphs at build
time. The final image runs those graphs through ONNX Runtime with Warp buffers;
source execution retains the original PyTorch backend when compiled graphs are absent.
A partial, changed, or mismatched compiled bundle fails verification.

`build-inputs.json` records source, model, resource, and file hashes;
`build-receipt.json` records the built image ID and expanded size.
Retain them when transferring an image with `docker save` / `docker load`.
Expanded size does not determine compressed transfer size or download duration.

Run source tests in the project environment with
`python -m unittest discover -s holoboxing/evaluation/tests`.
Build the player with `npm ci` and `npm run build` from `evaluation/viewer/`.
Review applicable model, robot, arena, and dependency licenses before redistribution.
Integrity verification does not grant redistribution rights; player notices
remain in [evaluation/viewer/notices](viewer/notices).

## Relation to prior work

The controller comes from the [HoloMotion project](https://github.com/HorizonRobotics/HoloMotion).
Its research reference is Chen et al.,
[HoloMotion-1 Technical Report](https://arxiv.org/abs/2605.15336), 2026.
That report describes V1; this evaluator uses the pinned V2 bundle and the
five-point interface documented above. Competition hit, fall, and latency rules
are evaluator definitions, rather than claims from that paper.

Zhou et al., [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035), CVPR 2019,
is the source for the rotation-representation method. The evaluator's exact
X/Z packing and coordinate frames remain part of this competition interface.
