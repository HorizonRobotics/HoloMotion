---
name: train-motion-policy
description: Train, fine-tune, evaluate, and export a HoloMotion motion-tracking policy from robot HDF5 data. Use when users want to fine-tune the released model on their own motions, train a policy, resume a run, evaluate checkpoints, export ONNX, or prepare a model for simulation and deployment.
---

# Train Motion Policy

Follow the complete model path:

```text
robot HDF5 -> training or fine-tuning -> offline evaluation -> ONNX export -> sim2sim
```

Read `docs/train_motion_tracking.md`, `docs/evaluate_motion_tracking.md`, and `docs/mujoco_sim2sim.md`. Use the project training environment through the `holomotion-train-interpreter` skill.

## Choose training or fine-tuning

- Prefer fine-tuning when the user has the released v1.4 checkpoint package and wants to adapt it to custom motion data.
- Use training from scratch only when requested or when the pretrained policy contract is incompatible.
- Verify that the checkpoint package contains the main checkpoint and its actor/critic state directories before fine-tuning.

Inspect the selected Hydra config rather than relying on remembered defaults. Confirm:

- every `train_hdf5_roots` path exists and sampling ratios are intentional;
- robot, DOF order, observations, actions, history/future reference, and network module agree;
- experiment name and output directory are distinct;
- resume and fine-tune semantics match the user's intent;
- requested environment count fits available memory.

## Preflight

Start with the smallest useful command or dry run. For the released fine-tuning entry:

```bash
HOLOMOTION_FINETUNE_DRY_RUN=1 \
  bash holomotion/scripts/training/finetune_motion_tracking_v1_4_0.sh
```

Review the printed command, checkpoint, config, dataset overrides, GPU selection, and output location before starting a long run.

Use:

- `holomotion/scripts/training/finetune_motion_tracking_v1_4_0.sh` for released-model fine-tuning;
- `holomotion/scripts/training/train_motion_tracking.sh` for the standard public training path.

Do not edit shared base configs merely to launch one experiment. Prefer a dedicated config or explicit Hydra overrides.

## Evaluate and export

Do not judge a policy from training reward alone.

1. Run offline evaluation with `holomotion/scripts/evaluation/eval_motion_tracking.sh`.
2. Inspect per-clip failures and generated reports, not only aggregate metrics.
3. Calculate offline metrics and visualize representative NPZ outputs.
4. Confirm that evaluation produces the expected ONNX artifact.
5. Run MuJoCo sim2sim before real-robot deployment when the required assets are available.

Keep training, evaluation, and deployment observation/action contracts aligned. If exported behavior differs from PyTorch evaluation, localize the first divergent stage before tuning the policy.

## Completion

Report the exact config, data roots, checkpoint lineage, command, outputs, evaluation evidence, exported model path, and unperformed deployment gates. Never claim real-robot success from simulation-only evidence.
