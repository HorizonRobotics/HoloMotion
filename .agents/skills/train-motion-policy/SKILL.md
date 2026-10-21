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

For boxing, resolve the participant training documentation separately from
motion-tracker training. Keep reference-only representation learning,
standing-opponent warmup, and self-play as distinct stages. Discover which
stages are implemented before offering a launcher. For a training handoff,
record rule/resource alignment, completed end-to-end execution, and externally
scored behavior as separate conclusions. A working legacy trainer does not
establish effectiveness after rules or robot assets change. Migrate those
contracts before carrying its success claims into the current evaluator.
A frozen decoder's good
reconstruction or encoded-trajectory tracking does not establish that arbitrary
latent samples or transitions are safe. Store pre-normalization samples and
their probabilities when optimizing a Gaussian projected onto a sphere.
Version dataset selection profiles and split original motion groups before
online augmentation. Verify batch-generated mirrors against independently
reflected source poses, and check normalization against the equivalent
balanced population without retaining duplicate training clips.
For sample training, start with representative small data and a compact model.
Keep the actor configuration consistent across supervised and reinforcement stages.
Measure runtime and held-out behavior before accepting a capacity reduction.
Choose architecture and early stopping with validation data.
Evaluate held-out test groups only after fixing the selected checkpoint.
Keep test clips and their mirrors outside pretraining and fitted statistics.

Keep a research opponent's native controller contract separate from the
candidate's submission contract. Use a dedicated training evaluator when those
interfaces differ. Preserve the robot, physics and referee rules. Freeze and
hash the opponent, then verify its observation, normalization and action mapping
against the original implementation. Qualify the candidate separately in the
unmodified competition runtime.
For reproduction, use the fixed competition reset on both candidate sides.
Keep temporary perturbation studies and repeated-match statistics in experiment records rather than the participant training workflow.
A source archive can omit generated observation graphs and select another backend.
Preserve the published runtime's verified compiled bundle for qualification.
Install the shared contact sampler before constructing a native-match scene.
Bind the referee to its measured physics substeps; frame aggregates omit impulse evidence.

A passive standing opponent should hold world targets if stationary behavior
is intended. Repeating a latent does not anchor the resulting physical pose.
Validate target invariance, physical drift, tilt, and contact response separately;
do not infer stability from frozen parameters alone.
Exclude inactive opponents from PPO updates and motion-discriminator negatives.
Freeze an opponent's observation statistics together with its policy weights.
Save historical snapshots, selection randomness and refresh progress for resume.
Change frozen opponents at update boundaries and reset physical episodes and histories.
Compute the simulated time between forced opponent resets from rollout length,
control period and refresh interval. Ensure this window exposes the intended
encounter duration; frequent snapshot refresh can truncate every fight.

For distributed training, distinguish environments per rank from the global
population and from inference batch size. Preserve each environment's recurrent
state when bounding expert-gather memory with inference batches. Synchronize
population statistics and gradients, then verify independent-row inference and
small-versus-batched physics before relying on large-scale learning results.
Use behavioral evaluation gates for curriculum changes rather than reward or
process completion alone; keep the frozen action contract across promotion.
Preserve invalid simulator states for diagnosis. If a training environment
terminates and resets such trials, count them separately, prevent terminal
bootstrapping and discriminator contamination, and exclude them from behavioral
acceptance. Do not turn invalid samples into successful hits or zero-cost steps.

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
Check cached previous-state fields when defining a current-frame observation.
Preserve the selected information contract through imitation, RL initialization and export.
When cleaning experimental entrypoints, retain the selected weights and fitted statistics.
Compare exported outputs before rerunning complete fixed-reset matches.
For mixed position/rotation outputs, report export errors in meters and radians.
Inspect pre-projection errors before attributing amplified FP32 rounding to changed weights.
Keep explicit physical tolerances, invalid-output checks and frozen-tensor identity checks.

## Completion

Report the resolved config, datasets, checkpoint lineage, command, topology,
outputs, evaluation coverage, exported artifact, and unperformed gates. Never
claim real-robot success from training or simulation evidence.
