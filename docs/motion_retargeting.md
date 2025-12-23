# Motion Retargeting

Transform human motion data into robot-compatible joint trajectories for following training. We support GMR for retargeting (https://github.com/YanjieZe/GMR)

## Prerequisites

Before running the motion retargeting pipeline, ensure you have:

### 1. Environment Setup

Please make sure the smplx and GMR are properly installed according to [[environment setup doc](docs/environment_setup.md)].

### 2. Data Preparation

Place your AMASS motion data in `/assets/test_data/motion_retargeting/{dataset_name}`
or modify 'amass_dir' in 'script/motion_retargeting/\*.sh' !Please check all related path in .sh and .yaml are right!

### 3. Model Preparation

    Put SMPLX models under following path
    thirdparties/
    └── GMR/
        ├── assets/
        │   └── body_models/
        │       └── smplx/
        │           ├── SMPLX_FEMALE.npz
        │           ├── SMPLX_FEMALE.pkl
        │           ├── SMPLX_MALE.npz
        │           ├── SMPLX_MALE.pkl
        │           ├── SMPLX_NETURAL.npz
        │           └── SMPLX_NETURAL.pkl

### 4. Path Verification

Check data paths in the configuration scripts:

- `holomotion/scripts/motion_retargeting/run_motion_retargeting_gmr.sh`

## Quick Start
### 1. Motion Retargeting

```bash
bash ./holomotion/scripts/motion_retargeting/run_motion_retargeting_gmr_smplx.sh
```

> Reminder: set device = "cuda:0" to "cpu" in "smplx_to_robot_dataset.py" if facing cuda error

After GMR retargeting, we further need to convert the dataset into a HoloMotion-compatible npz format, please run:

```bash
bash ./holomotion/scripts/motion_retargeting/run_motion_retargeting_gmr_to_holomotion.sh
```

### 2. Motion Visualization

Generate video outputs to validate retargeting quality of the HoloMotion npz files:

```bash
bash ./holomotion/scripts/motion_retargeting/run_motion_viz_mujoco.sh
```

**Output**: `video_rendering/{motion_name}.mp4` files in the retargeted data directories

### 3. Pack to HDF5 for Training
After retargeting, we need to pack the npz files into a compact HDF5 database:
```bash
bash ./holomotion/scripts/motion_retargeting/pack_hdf5_dataset.sh
```