# Boxing Planner Training

English | [简体中文](README.zh-CN.md)

Train a high-level boxing policy with a **frozen HoloMotion V2 controller**.
The planner predicts five-point pose targets; HoloMotion turns them into robot
joint actions. Start with supervised learning from reference motions, then
train an attacker against a humanoid sandbag in the competition ring.

**Samples:** The training methods, configurations, and commands are samples
for learning the workflow or using as a reference for your own development.

Training examples do not define the match referee.
Validate exported policies with the current [v10 match rules](../evaluation/README.md#match-rules)
and [observation contract](../evaluation/README.md#observations).
The 20–100 ms target horizon is separate from the referee's 20 ms hit window.

```text
Prepare data -> Supervised training -> Reinforcement learning -> Export -> Evaluate
```

## Environment and commands

Start with the HoloMotion [training environment](../../docs/environment_setup.md)
and an NVIDIA GPU. Create a separate environment for boxing training, then run
the following commands from the repository root:

```bash
conda create --name holoboxing --clone holomotion_train
conda activate holoboxing
export Train_CONDA_PREFIX="$CONDA_PREFIX"
"$Train_CONDA_PREFIX/bin/python" -m pip install -r holoboxing/training/requirements.txt
```

## Prepare resources and data

Download the [competition assets](../evaluation/README.md#source-resources)
to make `holoboxing/assets` available. Download the frozen lower controller:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing download --directory models/holomotion-v2
```

Follow the [data preparation guide](../data/README.md), using its designated
environments, to prepare the sample motions with the same
competition assets. The examples below use the combined dataset's output,
`holoboxing/data/processed/boxing/prepared/`.
If you prepared only RoboStriker or InterHuman, replace `--data` and `--amp-motions`
with that dataset's `prepared/` path.

| Input | What it contains | Argument |
| --- | --- | --- |
| `holoboxing/assets` | Competition robot and ring assets | `--assets` |
| `models/holomotion-v2` | Frozen HoloMotion controller | `--lower` |
| `holoboxing/data/processed/boxing/prepared` | Prepared reference motions | `--data` or `--amp-motions` |

Pass directories, not individual model or manifest files. Keep each bundle's
files together.

The same prepared directory serves two purposes:
- **Supervised learning:** reference states form the inputs and future reference
  poses form the targets.
- **AMP (Adversarial Motion Prior):** reference movements encourage natural
  motion during reinforcement learning, alongside the boxing rewards.

## Supervised training

```bash
CUDA_VISIBLE_DEVICES=0 "$Train_CONDA_PREFIX/bin/python" -m holoboxing.training.train_sl \
  --assets holoboxing/assets \
  --lower models/holomotion-v2 \
  --data holoboxing/data/processed/boxing/prepared \
  --output results/boxing-sl \
  --epochs 8 \
  --device cuda:0
```

`--epochs` sets the number of passes through the data. This example saves
`results/boxing-sl/epoch_000008.pt` for the next stage.

## Reinforcement learning

Initialize from the supervised model and train against the humanoid sandbag:

```bash
CUDA_VISIBLE_DEVICES=0 "$Train_CONDA_PREFIX/bin/python" -m holoboxing.training.train_rl \
  --assets holoboxing/assets \
  --lower models/holomotion-v2 \
  --amp-motions holoboxing/data/processed/boxing/prepared \
  --supervised-checkpoint results/boxing-sl/epoch_000008.pt \
  --output results/boxing-rl \
  --updates 40000 \
  --device cuda:0
```

`--updates` sets the number of PPO iterations.

## Export and evaluate

Export the trained attacker as a single ONNX file:

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.training.export \
  --checkpoint results/boxing-rl/model_40000.pt \
  --output results/boxing.onnx
```

Submit the exported `.onnx`, not the training `.pt` checkpoint.
Follow [HoloBoxing evaluation](../evaluation/README.md#run) to validate it, run a match
and watch the replay. With the evaluation image configured:

```bash
./holoboxing/run.sh --validate results/boxing.onnx
```

## Modify your planner

| Change | Start here |
| --- | --- |
| Policy network | [rsl_rl/modules/actor_critic.py](rsl_rl/modules/actor_critic.py) |
| Observations | [envs/mujoco_env.py](envs/mujoco_env.py) and [rsl_rl/modules/normalization.py](rsl_rl/modules/normalization.py) |
| Boxing and sandbag rewards | [tasks/combat/mdp/rewards.py](tasks/combat/mdp/rewards.py) and [envs/mujoco_rewards.py](envs/mujoco_rewards.py) |
| PPO, AMP and environment settings | [configs/training.json](configs/training.json) |

Keep the [competition policy interface](../evaluation/README.md#policy-interface):
`obs[1,1097]` in, `target_pose[1,225]` out, with the lower controller frozen.

## Methods and sources

This is a workflow example, not a reproduction of any cited paper's reported performance.

| Work | What is reused | What changes here |
| --- | --- | --- |
| Yin et al., [RoboStriker: Latent-Space Strategic Games for Autonomous Humanoid Boxing](https://arxiv.org/abs/2608.16195), 2026; [preview code](https://github.com/yinkangning0124/RoboStriker-Preview) | Sandbag-training structure, reward components, and PPO/AMP implementation structure | We jointly train the high-level task/prior/decoder network to output five-point pose targets, then execute them through frozen HoloMotion V2 |
| [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) | Frozen V2 tracking controller and shared observation/normalization utilities | Only the high-level boxing policy is trained; the tracker is not retrained |
| Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017; [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2016 | Clipped policy updates and advantage estimation | The local implementation handles paired boxing roles and high-level target actions |
| Peng et al., [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://xbpeng.github.io/projects/AMP/), 2021 | A motion discriminator supplies an additional style reward | The sample uses G1 transition features and RoboStriker-style binary cross-entropy training, alongside boxing task rewards |
| Schwarke et al., [RSL-RL: A Learning Library for Robotics Research](https://arxiv.org/abs/2509.10771), 2025; [code](https://github.com/leggedrobotics/rsl_rl) | The training modules retain this implementation lineage and notices | The bundled paired-role PPO/AMP modules are adapted code, rather than an unmodified upstream installation; see [license](rsl_rl/LICENSE) |
| Luo et al., [SMPLOlympics: Sports Environments for Physically Simulated Humanoids](https://arxiv.org/abs/2407.00187), 2024 | Discriminator initialization and gradient-penalty conventions recorded in the source | The sample trains competition G1 robots with local task features and rewards |

Both implementations use latent representations. In this sample, the task and
history-prior branches produce a normalized 32D latent. The jointly trained
decoder maps that latent and the current state to `target_pose[1,225]`.
PPO samples and scores the 225D target action; the internal latent is not its
stochastic action space. Frozen HoloMotion V2 converts these targets into joint actions.
RoboStriker's strategic policy instead selects latent actions for a pretrained motion decoder.
The internal bottleneck alone does not establish a training or inference speedup;
that requires a measured comparison.
The commands above cover supervised initialization and sandbag training; they do
not implement the paper's complete latent-space fictitious self-play pipeline.
They do not establish a performance comparison with RoboStriker.

Implementation links: [policy network](rsl_rl/modules/actor_critic.py),
[PPO](rsl_rl/algorithms/ppo.py), [advantages](rsl_rl/storage/rollout_storage.py),
[AMP](rsl_rl/modules/amp.py), and [discriminator](rsl_rl/modules/amp_discriminator.py).
For HoloMotion's research citation, see the
[HoloMotion-1 Technical Report](https://arxiv.org/abs/2605.15336), Chen et al., 2026.
That report describes V1; V2 behavior and interfaces follow the current code and model release.
