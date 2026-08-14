---
name: train-motion-policy
description: Train, fine-tune, resume, evaluate, and export a HoloMotion motion policy. Use for dataset selection, configuration preflight, checkpoint lineage, distributed training, evaluation coverage, export parity, and preparation for simulation or deployment.
---

# Train Motion Policy

Treat training as one contract-preserving pipeline:

```text
validated data -> training or fine-tuning -> evaluation -> export -> simulation -> deployment gates
```

Discover the current documentation, environment bootstrap, configuration
composition, entry points, evaluator, and exporter from the checked-out source.
Use `holomotion-train-interpreter` to run the selected environment.

## Choose training or fine-tuning

- Prefer fine-tuning when a compatible released or validated checkpoint exists.
- Use training from scratch only when requested or when the pretrained policy contract is incompatible.
- Verify checkpoint completeness and optimizer/model-state compatibility before fine-tuning or resume.

Inspect the fully composed training config rather than relying on remembered defaults. Confirm:

- every configured dataset exists, is immutable for the run, and has intentional sampling;
- robot, DOF order, observations, actions, history/future reference, and network module agree;
- experiment name and output directory are distinct;
- resume and fine-tune semantics match the user's intent;
- requested environment count fits available memory.

## Preflight

Resolve the current launcher from documentation and source. Prefer its
non-submitting dry-run or command-print mode; otherwise inspect it and run the
smallest local configuration that validates composition and startup. Review
the resolved command, checkpoint, config, data overrides, topology, and output
location before starting a long run.

Do not edit shared base configs merely to launch one experiment. Prefer a
dedicated config or explicit overrides supported by the current configuration
composer.

## Evaluate and export

Do not judge a policy from training reward alone.

1. Resolve the evaluator and export contract associated with the selected
   training config rather than assuming a script or flag name.
2. Require complete intended checkpoint, dataset, and clip coverage.
3. Inspect per-clip failures and balanced aggregates, not only a global mean.
4. Use machine-readable reports for exact values and human-readable summaries
   for review; discover their current schemas from the evaluator.
5. Confirm exported inputs, outputs, state, preprocessing, and numerical
   behavior against the training implementation.
6. Run the currently supported simulation evaluation before real-robot use.

Keep training, evaluation, and deployment observation/action contracts aligned. If exported behavior differs from PyTorch evaluation, localize the first divergent stage before tuning the policy.

## Completion

Report the resolved config, datasets, checkpoint lineage, command, topology,
outputs, evaluation coverage, exported artifact, and unperformed gates. Never
claim real-robot success from training or simulation evidence.
