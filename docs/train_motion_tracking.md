## Train the Motion Tracking Model

After [environment setup](environment_setup.md) and
[motion retargeting](motion_retargeting.md), run the following commands from
the repository root. Training is optional when using a compatible pretrained
policy; see [evaluation](evaluate_motion_tracking.md) or
[robot deployment](realworld_deployment.md) to use an existing model.

**Overall Workflow:**

```mermaid
flowchart LR
A[Motion Retargeting] --> B[HDF5 Database]
B --> C[Training Config]
C --> D[Training Entry]
D --> E[Distributed PPO Training]

classDef dashed stroke-dasharray: 5 5, rx:10, ry:10, fill:#c9d9f5
classDef normal fill:#c9d9f5, rx:10, ry:10
class A dashed
class B,C,D,E normal
```

### 1. Train v2.0 from scratch

Use the single v2.0 recipe:
`holomotion/config/training/motion_tracking/motion_tracking_v2_0_0.yaml`.
It composes the shared PPO, robot, observation, reward, terrain, and network
groups; these reusable groups are not separate experimental recipes.

The recipe uses rough terrain, 8,192 environments, an 8,192-clip motion cache,
five future reference frames, six named sparse-control modes, world-frame
tracking rewards, adaptive source sampling (S1), and same-motion failure replay
(S2). Every data source is eligible for every named mode. Distributed process
count comes from Accelerate and is not fixed to one GPU topology in the YAML.

The training configuration defines six reference masks with a fixed sampling
probability of 1/6 throughout
training. S1 adapts data-source probabilities and within-source motion-window
weights, not control-mode weights. S2 keeps the failed mode when replaying a
motion by default, so observed mode frequencies need not be exactly uniform.

The released teleoperation workflow exposes Pelvis-1 and Keypoint-5; the other
training masks remain experimental.

#### 1.1 Configure your training data

Replace `train_hdf5_roots` with your robot root/DOF HDF5 datasets. Set
`val_hdf5_roots` separately if you have held-out training validation data;
by default it uses the training roots. Training validation is separate from
independent model evaluation. See the
[HDF5 format and source metadata](holomotion_motion_file_spec.md#compact-rootdof-hdf5).

The default adaptive sampler requires a `metadata.source_id` on every eligible
clip and a complete `robot.motion.source_sampling.probabilities` mapping. The
example HDF5 path is not a full training dataset and is not guaranteed to contain
source labels. The probabilities field is deliberately required (`???`): fill
it with the source IDs actually present in your data before launching training.
Internal data, storage addresses, and experiment names are not included in this
public recipe.

For example, data labeled `source_a` and `source_b` can use:

```yaml
train_hdf5_roots:
  - root: data/my_training_dataset
    ratio: 1.0
robot:
  motion:
    source_sampling:
      probabilities:
        source_a: 0.6
        source_b: 0.4
```

The keys must exactly cover the eligible data sources. S1 adjusts their initial
training-window sampling probabilities within configured bounds; these are
not dataset sizes, clip-count ratios, or permanent environment assignments.
S2 retries failed segments of the same motion and also affects observed counts.
Neither finite random sampling nor the configured source ratio guarantees that
every clip has already appeared in a batch.

#### 1.2 Start training

After configuring data, run this command from the repository root:

```shell
bash holomotion/scripts/training/train_motion_tracking.sh
```

The script defaults to `motion_tracking_v2_0_0`. It preserves the recipe's
environment count unless `NUM_ENVS` is explicitly set. CUDA devices select
the local Accelerate topology; use your distributed launcher for multi-node
training. Optional environment and Hydra overrides remain available:

```shell
CUDA_VISIBLE_DEVICES=0,1 NUM_ENVS=4096 \
  bash holomotion/scripts/training/train_motion_tracking.sh \
  experiment_name=my_training_run base_dir=logs

# Print the command without starting training.
HOLOMOTION_TRAIN_DRY_RUN=1 \
  bash holomotion/scripts/training/train_motion_tracking.sh
```

The recipe saves every 500 iterations. Iterations are zero-indexed, so
`num_learning_iterations: 18001` includes saving `model_18000`.
Large diagnostic motion-key and whole-window-score dumps are off by default;
source/replay metrics and sampling checkpoint state remain enabled.

For historical training recipes, check out the corresponding release tag;
this branch exposes only the v2.0 training recipe.

Isaac Lab may download assets from NVIDIA's cloud storage. If scene creation
stalls, inspect asset-download connectivity before changing the training recipe.

### 2. Resume or fine-tune a compatible checkpoint

Keep the saved training configuration and the complete checkpoint together:

```text
run/
├── config.yaml
├── model_500.pt
├── model_500/actor/
├── model_500/critic/
└── exported/model_500.onnx
```

The `.pt` file stores optimizer, iteration, and sampling state; actor and critic
weights live in the adjacent directories. An ONNX export alone cannot be used
to resume or fine-tune training. The configured actor/critic architecture and
observation schema must match because weights are loaded strictly.

To resume the same training setup:

```bash
bash holomotion/scripts/training/train_motion_tracking.sh \
  checkpoint=/path/to/run/model_500.pt \
  ++algo.config.checkpoint_load_mode=resume
```

Resume restores the iteration and compatible optimizer state, and requires
matching source-adaptive sampling state. When adapting to a new dataset,
configure its roots and source probabilities, then select fine-tuning:

```bash
bash holomotion/scripts/training/train_motion_tracking.sh \
  checkpoint=/path/to/run/model_500.pt \
  ++algo.config.checkpoint_load_mode=finetune \
  experiment_name=my_finetune
```

Fine-tuning loads actor/critic weights with fresh optimizers and sampling state,
and starts at iteration zero by default. `checkpoint_load_mode` controls this
choice; the legacy `load_optimizer` field is not the switch used by the loader.
Evaluate the resulting exported bundle with the
[evaluation guide](evaluate_motion_tracking.md).

### Training Tips

#### How to use less GPU ?

Training requires significant GPU memory. Reduce `NUM_ENVS` if GPU memory is limited. This reduces both rollout and PPO memory consumption, but may make policy optimization less stable.

#### How to start multiple training session ?

Set `CUDA_VISIBLE_DEVICES` for each session. For simultaneous distributed jobs,
choose a distinct nonzero Accelerate port. The shell wrapper forwards trailing
arguments to Hydra, so pass launcher options by calling Accelerate directly:

```bash
source train.env
CUDA_VISIBLE_DEVICES=0,1 "$Train_CONDA_PREFIX/bin/accelerate" launch \
  --multi_gpu --main_process_port=29501 \
  holomotion/src/training/train.py \
  --config-name=training/motion_tracking/motion_tracking_v2_0_0 \
  headless=true experiment_name=my_training_run
```

#### How to set the save/log intervals ?

You may want to have more or less frequent logging and model dumping intervals. You can alter these intervals by adding the following options:

- `algo.config.save_interval=X` : The checkpoint will be saved every `X` learning iterations.
- `algo.config.log_interval=Y`: The logging information will be displayed every `Y` learning iterations.

#### Where is the checkpoint dumped ?

By default, checkpoints are written to `logs/<project_name>/<timestamp>-<experiment_name>/`. Change `base_dir`, `project_name`, or `experiment_name` to customize this path.
