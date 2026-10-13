# Sim2Sim Verification

Use MuJoCo to inspect an exported policy on your own retargeted reference
motion before considering real-robot deployment. No fixed benchmark dataset
is required.

From the HoloMotion repository root, with the training environment installed,
download the complete [v2.0.0 motion-tracking bundle](https://huggingface.co/HorizonRobotics/HoloMotion_models/tree/main/HoloMotion_motion_tracking_model_v2.0.0)
into `checkpoints/HoloMotion_motion_tracking_model_v2.0.0/`:

```bash
ONNX_PATH=checkpoints/HoloMotion_motion_tracking_model_v2.0.0/exported/model_10500.onnx \
MOTION_NPZ_PATH=/path/to/retargeted_motion.npz \
  bash holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh
```

Keep the matching saved training `config.yaml` in the model's run directory.
`ROBOT_XML_PATH` optionally overrides the default G1 29-DOF scene.
`HEADLESS=true` selects headless recording; `RECORD_VIDEO=false` disables
recording. Trailing Hydra overrides select `link_eval` control points.

For example, append `link_eval.mask_mode=custom` and
`'link_eval.active_link_names=[pelvis]'` to inspect pelvis-only control.
See the [evaluation guide](evaluate_motion_tracking.md) for control selection,
quantitative HDF5 evaluation and output definitions.

Watching a motion does not produce a complete dataset score, nor does
successful simulation establish real-robot safety.
