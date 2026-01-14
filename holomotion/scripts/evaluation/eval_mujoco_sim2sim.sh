source train.env

export CUDA_VISIBLE_DEVICES="0"

export HEADLESS=false
if $HEADLESS; then
    export MUJOCO_GL="osmesa"
    export RECORD_VIDEO=true
else
    export MUJOCO_GL="egl"
    export RECORD_VIDEO=false
fi

robot_xml_path="assets/robots/unitree/G1/29dof/scene_29dof.xml"

ONNX_PATH="logs/HoloMotionMotionTracking/xxxxx/exported/model_xxx.onnx"
export motion_npz_path="data/holomotion_retargeted/processed_datasets/AMASS_test/clips/ACCAD_Male1Walking_c3d_Walk_B10_-_Walk_turn_left_45_stageii.npz"

${Train_CONDA_PREFIX}/bin/python holomotion/src/evaluation/eval_mujoco_sim2sim.py \
    +ckpt_onnx_path="$ONNX_PATH" \
    record_video=$RECORD_VIDEO \
    headless=$HEADLESS \
    camera_tracking=true \
    camera_distance=7.0 \
    +motion_npz_path='${oc.env:motion_npz_path}' \
    +robot_xml_path=${robot_xml_path}
