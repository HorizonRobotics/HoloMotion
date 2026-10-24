# SuperMimic training in HoloMotion

English | [简体中文](supermimic.zh-CN.md)

**Train a lightweight G1 motion tracker from scratch on one CUDA GPU.**
Use HoloMotion's HDF5 motions, G1 assets, and training workflow with MuJoCo Warp or Isaac PhysX.
The selected MuJoCo recipe achieved **80% success: 32 of 40 complete LAFAN1 training clips after 30 minutes on one RTX 5090**.
See the [results and recipe](#7-recorded-30-minute-results) below.

<div align="center">
  <img src="media/supermimic_training_progress.webp" alt="HoloMotion training progression: PPO left, SuperMimic right; MuJoCo Warp above Isaac PhysX" width="640">
</div>

Rows: MuJoCo Warp and Isaac PhysX. Columns: PPO and SuperMimic.
The 60-second animation shows an independent training run from 0 to 30 minutes in three-minute stages, at the original motion speed.
Each panel replays the same first five seconds with native physics and continuous policy actions, including after falls. This illustration is separate from the full-clip success measurements below.

## Why use SuperMimic in HoloMotion?

Use this route when you have G1 reference motions and want to train or adapt a compact tracker on one GPU.
SuperMimic aims to obtain more policy-learning signal from each expensive physical sample by learning a differentiable world model.

### Reuse physical data for policy learning

![PPO and SuperMimic training loops: data flow and policy-gradient paths](media/supermimic_training_loop_en.png)

PPO reuses collected trajectories across optimization epochs, updating its policy through action log-probabilities and detached advantages.
SuperMimic also fits a world model from real transitions in replay, then uses it to generate new multi-step trajectories under the current policy.
Model fitting updates the world-model parameters φ.
Policy optimization freezes φ while retaining gradients through its state and action inputs, analytic rewards, and terminal values; the physical simulator does not participate in backpropagation.

Moving many policy-training rollouts from contact solving to batched neural-network computation can reduce compute per learning signal.
Model fitting and imagined backpropagation still cost compute, so model size, horizon, batch size, prediction quality, and hardware determine the actual benefit.
The [30-minute results below](#7-recorded-30-minute-results) measure real training reward over wall time for the recorded configurations.

### Reuse the HoloMotion workflow across simulators

- **Use existing data and assets.** Train through HoloMotion's HDF5 reader, original G1 assets, Hydra configuration, and training entrypoint.
- **Compare methods and backends.** SuperMimic and compact-policy PPO share motion data, observations, controller targets, rewards, and complete-clip scoring on MuJoCo Warp and Isaac PhysX.
- **Inspect each experiment.** Resolved configurations, source/input fingerprints, checkpoints, training reward curves, and raw evaluation transitions make changes auditable.

This integration adapts the compact upstream recipe to HoloMotion’s HDF5 motions, G1 assets, Hydra commands, and two simulator backends.
It adds shared task and evaluation interfaces, training curves, recoverable learning state, and source/input checks.
See [attribution](#attribution-and-implementation-notes) for the upstream implementations.

### Current scope and roadmap

The validated actor has about 9M parameters; this compact scope makes the complete training loop practical to test on one GPU.
A large-model SuperMimic version is planned, including integration with HoloMotion's Transformer, MoE, and temporal-state contracts.
An efficiency benefit for those large policies has not yet been demonstrated and will need its own measurements.
For an existing pretrained sparse-to-whole-body controller, use the [HoloMotion v2 guide](../README.md#holomotion-v2-flexible-sparse-to-whole-body-command-following-with-cross-modal-extensibility).

## 1. Install a separate environment

Use Python 3.11, an NVIDIA CUDA GPU, and a compatible driver.
The recorded experiments used RTX 4090 and RTX 5090 GPUs.
A minimum supported GPU-memory capacity has not yet been established.

Run these commands from the repository root:

```bash
python3.11 -m venv .venv-supermimic
source .venv-supermimic/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements_supermimic.txt
python -m pip install --no-deps -e .
python -m pip check
```

This MuJoCo Warp recipe does not initialize Isaac Sim or Isaac Lab.
Keep the recipe in the virtual environment created above.

For Isaac PhysX, use the repository's Isaac Sim 5.0 and Isaac Lab 2.2 environment.
Do not install the MuJoCo Warp requirements over that environment's simulator dependencies.
Select Isaac with `algo.config.physics_backend=isaac_physx`.
Its launcher runs training and final evaluation in separate supervised processes.
Isaac supports fresh training and final evaluation; periodic evaluation and checkpoint recovery are currently unavailable.

Check the selected runtime:

```bash
python -c 'import torch, mujoco, warp; print(torch.__version__, mujoco.__version__, warp.__version__); print(torch.cuda.is_available())'
```

The expected physics versions are MuJoCo 3.16.0 and Warp 1.18.0.
The default network arithmetic uses BF16; physical state, rewards, and normalization use FP32 tensors.

## 2. Prepare the reference motions

The reference dataset contains 40 complete G1 LAFAN1 CSV clips at 30 Hz.
Each row stores root XYZ, an XYZW quaternion, and 29 SDK-order joint angles.
Download the `g1/` CSVs from [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
The recorded dataset revision is `ce1572906efe6157840e8474d5a0d7aa87481e74`.
Motion data retains the terms published by its source and is not redistributed with this implementation.

One download option is the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
python -m pip install huggingface-hub
hf download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset \
  --revision ce1572906efe6157840e8474d5a0d7aa87481e74 \
  --include 'g1/*.csv' --local-dir data/lafan1-source
```

Convert the complete clips into HoloMotion HDF5 data:

```bash
python -m holomotion.scripts.training.prepare_supermimic_lafan1 \
  --source data/lafan1-source/g1 --output data/lafan1-g1-hdf5
```

The converter resamples poses to 50 Hz and writes `manifest.json`, `motions.h5`, and source fingerprints.
It retains complete clips; it does not create a held-out test split.
The standard HoloMotion reader then applies its usual initial XY and yaw normalization.

The robot asset is `assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf`.
Keep its referenced mesh files available alongside the URDF.
The recipe generates its own MuJoCo model in the experiment directory without editing the original asset.

## 3. Check physics before training

```bash
CUDA_VISIBLE_DEVICES=0 python -m holomotion.src.training.train \
  --config-name=training/motion_tracking/supermimic \
  motion_h5_path=data/lafan1-g1-hdf5/manifest.json \
  robot_urdf=assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf \
  experiment_name=supermimic-preflight \
  num_envs=2 algo.config.preflight_only=true \
  algo.config.warmup_steps=4 algo.config.capture_graph=false
```

A completed `training/run_status.json` establishes physical execution only.
It does not establish policy learning or tracking quality.
Capacity overflow and nonfinite states reject a run and retain diagnostic evidence.

## 4. Train on one GPU

Start a fresh run with the selected 30-minute MuJoCo recipe:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync CUDA_VISIBLE_DEVICES=0 \
python -m holomotion.src.training.train \
  --config-name=training/motion_tracking/supermimic_30min \
  motion_h5_path=data/lafan1-g1-hdf5/manifest.json \
  robot_urdf=assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf \
  experiment_name=supermimic-30min
```

This recipe uses horizon 24, learning rate `4e-4`, and a full 4,096-sample actor microbatch.
It trains for 1,800 seconds and evaluates the final policy; periodic evaluation is disabled within that budget.
The matching Isaac recipe uses `algo.config.physics_backend=isaac_physx`; omit the MuJoCo allocator prefix in the Isaac environment.
Its complete training and evaluation metrics are reported below.
The original `training/motion_tracking/supermimic` configuration remains available for reference recipes and smaller microbatches.

The default `paper` profile uses 93 proprioceptive values, ten history frames, and ten future reference frames.
The `vibed` profile adds six relative-orientation values and disables MuJoCo self-collisions.
Isaac keeps self-collisions enabled for both observation profiles.
Keep the profile label when comparing results; `paper` does not certify exact reproduction of every unspecified paper detail.

The recipe alternates real collection, world-model fitting, imagined policy updates, and critic fitting.
Resolved specifications record batch sizes, warmup, update rates, and measured timing.
An update can finish after the requested deadline; compare recorded training budgets.

Reduce `algo.config.pi_microbatch` first when actor-update memory is limiting.
Microbatches split one logical policy update; they do not change its intended sample count.
Changing `num_envs`, logical batch sizes, or update rates changes sample reuse and requires a separately labelled recipe.
The selected recipe already uses `algo.config.pi_microbatch=4096` to process the full logical actor batch.

By default, `algo.config.release_phase_cache=true` checks available GPU memory before physics launches and releases unused Torch cache only under pressure.
The policy uses actual device capacity and a conservative reserve for independent physics allocations.
It preserves live tensors; its reserve is a safety margin, not a measured transient-memory requirement.
Disable it only after checking memory headroom for your workload.

For MuJoCo, prefix the command with `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` to select Torch's asynchronous allocator.
Verify the effective allocator in `specification.json` and cache decisions in `report.json`.
`TORCHINDUCTOR_CACHE_DIR` and `WARP_CACHE_PATH` select writable compiler caches; compilation counts toward the training budget.

## 5. Inspect outputs

Experiments are stored under `logs/motion_tracking/<timestamp>-<experiment_name>/` by default.
The root contains the resolved framework configuration; `training/` contains the training artifacts.

| Artifact | Purpose |
| --- | --- |
| `specification.json` | Resolved options, input/source hashes, runtime, robot, and task contract. |
| `run_status.json` and `report.json` | Completion state, actual budgets, optimizer counts, physics status, and evaluation summaries. |
| `checkpoint_final.pt` | Models, optimizers, normalizers, replay, curriculum, update clock, and random-generator states. |
| `evaluation_final.json` | Final complete-clip scores when evaluation is enabled. |
| `rollout_final.pt` | Recorded physical evaluation transitions; Isaac stores this under `evaluation/`. |
| `world_diagnostic_final.json` | World-model diagnostics with their stated normalization. |
| `training_reward_curve.json` | Mean real step reward across training environments, with a training-wall timestamp for each vector step. |
| `policy_progress/` | Optional policy snapshots for visualizing learning; enable with `algo.config.policy_snapshot_minutes=3`. |
| `training_reward_area.json` | Reward area over the first 1,800 training seconds and the final 300-second mean, when the full horizon is reached. |

The training curve holds each measured real reward until the next observation and uses zero before the first observation.
Its clock includes training preparation and compilation, excludes independent evaluation, and ends before post-training finalization.
Short runs retain a partial area with `complete=false`; they do not receive a complete 30-minute time-mean reward.
These sampled training rewards describe a different population from the fixed complete-clip evaluation below.

Evaluation starts every clip from its first frame and uses deterministic policy actions.
The normalized return divides by the complete planned duration, including unexecuted failure tails.
Do not compare that score directly with an executed-steps-only average.

Resume a new-format checkpoint into a fresh experiment directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m holomotion.src.training.train \
  --config-name=training/motion_tracking/supermimic \
  motion_h5_path=data/lafan1-g1-hdf5/manifest.json \
  robot_urdf=assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf \
  experiment_name=supermimic-resumed \
  checkpoint=/path/to/previous/training/checkpoint_final.pt \
  algo.config.max_hours=1.0
```

Keep the original data paths and recipe overrides when resuming.
The trainer validates source, inputs, model dimensions, and update scheduling before restoring learning state.
Budgets are cumulative: `max_hours=1.0` continues toward one total training hour.
Recovery restores learning state but starts fresh physical episodes; it does not restore simulator internals.
MuJoCo supports this recovery; Isaac and PPO currently support fresh training only.
Earlier inference-only bundles cannot resume training.
Load only trusted checkpoints.

## 6. Run a PPO control on either simulator

The simulator PPO control shares the compact policy, critic, motion data, and task with the model-based recipe.
It learns from physical rollouts using clipped PPO updates and a learned Gaussian exploration distribution.
The default recipe uses 4,096 environments, 24-step rollouts, five epochs, and four logical minibatches.

Use the selected simulator's environment from the installation section:

```bash
CUDA_VISIBLE_DEVICES=0 python -m holomotion.src.training.train \
  --config-name=training/motion_tracking/reference_ppo \
  motion_h5_path=data/lafan1-g1-hdf5/manifest.json \
  robot_urdf=assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf \
  algo.config.physics_backend=mujoco_warp \
  experiment_name=reference-ppo-mujoco
```

Select `algo.config.physics_backend=isaac_physx` and a fresh experiment name for the Isaac control.
Both defaults train for 30 minutes, then run complete-clip evaluation.
Both methods write the training reward artifacts described above and use the same complete-clip evaluator.
The direct CLI is `python -m holomotion.scripts.training.train_reference_ppo --help`.
Compare within each simulator using identical inputs, initialization, hardware, precision, and training budgets.

## 7. Recorded 30-minute results

Each run trained from scratch with seed 0, 4096 environments, one RTX 5090, and 1800 seconds of training wall time.
The selected SuperMimic recipe uses horizon 24 and learning rate `4e-4`; its MuJoCo policy achieved **32/40, or 80% SR**.
The matching Isaac run achieved **24/40, or 60% SR**.

| Method | Simulator | Training reward area ↑ | Time-mean training reward ↑ | Final evaluation reward ↑ | Success (40 clips) |
| --- | --- | ---: | ---: | ---: | ---: |
| SuperMimic | MuJoCo Warp | **73.316052** | **0.040731** | **0.465671** | **32/40 (80%)** |
| PPO baseline | MuJoCo Warp | −20.218714 | −0.011233 | 0.003744 | 0/40 |
| SuperMimic | Isaac PhysX | **78.286892** | **0.043493** | **0.421147** | **24/40 (60%)** |
| PPO baseline | Isaac PhysX | −30.728645 | −0.017071 | 0.003401 | 0/40 |

![Real training reward for selected SuperMimic and reference PPO on both simulators](media/supermimic_training_reward_20261024.png)

Training reward area integrates the **real physical training reward** over the first 1,800 seconds, in reward-seconds.
It includes task setup and compilation, holds each observed mean step reward until the next observation, and uses zero before the first observation.
Final checkpoint writing and evaluation are excluded. Negative rewards remain negative.
The figure shows exact 30-second time means, with faint raw observations; its areas use the full recorded curves.
This is not imagined-rollout reward or an area under evaluation scores.

SuperMimic and PPO share the compact policy and critic architecture, initial actor/critic tensors, reward, motion inputs, observations, and controller contract.
PPO learns from real rollouts without a world model; its learned exploration standard deviation starts at 1.0.
SuperMimic uses fixed collection noise 0.1.
The table compares the selected SuperMimic recipe with reference PPO runs, not independently tuned best configurations for both algorithms.
The selected MuJoCo recipe prioritizes final success; higher success does not imply improvement in every training metric.
These single-seed results do not establish cross-seed stability or an algorithm-wide advantage.

Final evaluations use each simulator's original native protocol, on RTX 5090, with the same 40 complete training-set clips and 441100 planned transitions.
Success means completing a clip without the height or tilt termination rule; it does not impose a global tracking-error threshold.
Normalized evaluation reward includes the complete planned duration, including unexecuted failure tails.
PhysX and MuJoCo differ in contact dynamics and motor-velocity constraints.
No held-out generalization or real-robot transfer is established here.
The opt-in canonical MuJoCo evaluator is a separate protocol and is not used for this table.

The [machine-readable results](supermimic_results.json) provide configurations, source versions, and verified metric provenance.

## Optional canonical MuJoCo evaluation

Saved compact SuperMimic and PPO policies can use an explicit canonical GPU evaluation mode in the MuJoCo environment:

```bash
python -m holomotion.scripts.evaluation.eval_reference_policy \
  --checkpoint outputs/run/training/checkpoint_final.pt \
  --output outputs/run/canonical-evaluation \
  --mode canonical_gpu_v1
```

The command starts a dedicated evaluation process, verifies the checkpoint's shared task and input fingerprints, saves raw transitions, and records completion and timing.
Use `--manifest` and `--robot-urdf` to relocate inputs with identical verified contents.
The mode pins dependency sources and the supported paper-profile G1 model/options; model fitting and simulator training retain their original implementation.

A fixed policy produced identical full 40-clip graph/eager transition traces on one GPU with canonical ordering.
Reproducibility across devices and hosts remains unqualified.

Canonical ordering changes floating-point trajectories, so its scores require a separate baseline.
The command defaults to `--mode native`; the results above and existing training commands use native evaluation.

## Attribution and implementation notes

The shared task lives in `holomotion/src/env/reference_tracking/`; complete-clip scoring lives in `holomotion/src/evaluation/reference_tracking/`.
The SuperMimic package owns learned dynamics, replay, imagined updates, and its training lifecycle.
The policy encodes future reference motion into a bounded 32-value latent and combines it with ten frames of proprioceptive history.
The world model predicts local state increments; a critic supplies terminal values and fits lambda-return targets.

The analytic reward covers body pose, body velocity, and root pose, with penalties for action changes and soft joint-limit violations.
It is shared by imagined and physical states.
The controller applies joint position targets at 50 Hz with four 5 ms physics steps.
MuJoCo Warp and Isaac PhysX implement this contract with their respective contact dynamics.

Credit the [SuperMimic paper](https://arxiv.org/abs/2610.28480) for the method.
The compact network and update recipe adapts [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) at the revision recorded in [`upstream.json`](../holomotion/src/algo/supermimic/upstream.json).
Its [MIT notice](../holomotion/src/algo/supermimic/LICENSE.vibed) is retained.
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) provided an independent cross-check.
