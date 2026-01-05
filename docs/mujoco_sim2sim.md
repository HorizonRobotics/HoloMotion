# Sim2Sim Verification

After generating the ONNX file from the evaluation stage, you can verify the performance of your Isaac-trained policy in another simulator, such as Mujoco to test its performance before deploying to the real robot.

The entry script is `holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh`, you should set these variables before running:

- `robot_xml_path`: The scene mjcf .xml file for the robot
- `ONNX_PATH`: The exported ONNX model file
- `motion_pkl_path`: The npz file containing the reference motion
