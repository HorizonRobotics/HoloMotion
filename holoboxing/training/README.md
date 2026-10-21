# Boxing training reference

Follow the [participant tutorial](../README.md#1-train-the-high-level-policy) for data preparation, the three training stages, export, and competition validation.
This reference covers checkpoint reuse and a fixed native-opponent development test.
It does not change the submission interface or competition rules.

## Preserve the trained policy

For reproducing an existing result, start from the selected `.pt` checkpoint or its verified ONNX export.
Keep the actor, frozen decoder, and observation statistics together.
Record the checkpoint, exported model, data manifests, controller, resources, source revision, runtime image, and opponent hashes.
The same hyperparameters with different data or weights do not reproduce a selected policy.

The current planner uses current self state and opponent observations.
The 1097-value public input is unchanged; older history and the previous measured pose are ignored inside the planner.
New frozen-prior training uses this contract automatically.
The loader retains the earlier full-history checkpoint contract; it does not reinterpret an old checkpoint as a current-state policy.

`sample.json` fixes the actor widths and standing-stage settings.
`selfplay.json` keeps that architecture and sets the new stage's initial exploration standard deviation to 0.2.
Neither preset modifies the competition controller or simulation rules.

## Resume versus a new training stage

`train_sl --resume` restores the embedded decoder, optimizer, random state, and next batch.
Supply the same data, actor configuration, batch size, learning rate, and training seed.
No original prior file is needed for supervised resume.
`--epochs` is the total target epoch count, including completed epochs.

`train_rl --resume` restores the learner, optimizer, normalizers, replay state, and historical opponent pool.
Physical episodes and observation histories restart.
`--updates` requests additional updates; subtract completed updates when targeting a particular final checkpoint number.
Choose a new output directory for each invocation.

`train_rl --policy-checkpoint` instead starts a new curriculum stage from the saved actor and actor normalizer.
It initializes new critics, optimizers, exploration, and opponent-pool state.
Do not substitute resume for the standing-to-self-play transition.

Check `training-result.json` for actual completed progress and the saved checkpoint hash.
Reaching a wall-time limit is not successful completion of the requested update budget.

## Fixed native RoboStriker development opponent

The original RoboStriker policy outputs joint actions; a competition submission outputs five-point target poses.
The dedicated `evaluate_native` adapter preserves the opponent's original inference interface while the candidate still uses the public observation and frozen HoloMotion controller.
It uses the competition arena, measured contact substeps, controller gains, and referee.
This is a development comparison, not the official two-ONNX submission evaluator or cloud qualification.

Obtain the G1 Stage-3 checkpoint from [RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview).
Use its original `model_3000.pt`, and retain the accompanying license.
Freeze that file before comparing candidates; `--opponent-sha256` must match its SHA256 digest.

Run two complete matches from the exact competition initial state:

```bash
export ROBOSTRIKER_CHECKPOINT=/absolute/path/to/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
python -m holoboxing.training.evaluate_native \
  --candidate results/boxing.onnx \
  --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
  --opponent-sha256 "$ROBOSTRIKER_SHA256" \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --candidate-side 0 --output results/native-side0 --device cuda:0
python -m holoboxing.training.evaluate_native \
  --candidate results/boxing.onnx \
  --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
  --opponent-sha256 "$ROBOSTRIKER_SHA256" \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --candidate-side 1 --output results/native-side1 --device cuda:0
```

Both roles use the supplied reset pose without random initial-position perturbations.
`result.json` identifies the candidate side, model hashes, completion status, score, and referee winner.
`match.json` retains per-frame states and events.
Judge wins from the referee result; hits, falls, and KO rewards are distinct evidence.
A win does not necessarily mean more valid hits or no candidate falls.

For comparison with a published image, preserve its compiled runtime bundle, original resources, and dependency versions.
Run the external native-opponent adapter against that runtime; do not replace image source files or relabel a source-backend result as compiled-runtime evidence.
Report the runtime image and adapter separately.
Use the normal tutorial Docker commands for candidate qualification and official two-ONNX matches.
