# Motion Retargeting


Transform human motion data into robot-compatible joint trajectories for following training.


## Overview

The motion retargeting pipeline consists of three main components:

1. **Shape Fitting** - Optimize SMPL shape parameters to match target robot morphology
2. **Motion Retargeting** - Convert human motion to robot joint trajectories using PHC or Mink optimization
3. **Result Visualization** - Generate video outputs to validate retargeting quality

## Prerequisites

Before running the motion retargeting pipeline, ensure you have:

1. **Environment Setup**: Set the `holomotion_train` conda environment
   ```bash
   conda env create -f environment_train.yaml
   conda activate holomotion_train
   ```

2. **Data Preparation**: Place your AMASS motion data in `/assets/test_data/motion_retargeting/{dataset_name}`
   or modify 'amass_dir' in 'script/motion_retargeting/*.sh' !Please check all related path in .sh and .yaml are right!

3. **Model Preparation**: Beside SMPL*.pkl file, please download three SMPLX*.npz model files from 'https://smpl-x.is.tue.mpg.de/download.php'.  
​   ​Download from the first option under 'SMPL-X MODEL'  
   Then put 'SMPLX_NETURAL.npz', 'SMPLX_MALE.npz', 'SMPLX_FEMALE.npz' at `/assets/smpl/*`

4. **Path Verification**: Check data paths in the configuration scripts:
   - `holomotion/scripts/motion_retargeting/*.sh`

## Quick Start

### 1. Shape Fitting

Fit SMPL shape parameters to your target robot morphology:

```bash
cd {holomotion_path}
bash ./holomotion/scripts/motion_retargeting/run_robot_smpl_shape_fitting.sh
```

**Output**: `fitted_smpl_shape.pkl` file at `assets/robots/unitree/G1/21dof/`

**Configuration**: Modify `shape_fitting_config.yaml` for custom robot specifications and reference XML files.

### 2. Motion Retargeting

Choose between two retargeting methods:

#### Option A: PHC Retargeting
```bash
cd {holomotion_path}
bash ./holomotion/scripts/motion_retargeting/run_motion_retargeting_phc.sh
```

#### Option B: Mink Retargeting
```bash
cd {holomotion_path}
bash ./holomotion/scripts/motion_retargeting/run_motion_retargeting_mink.sh
```

**Output**: `{motion_name}.pkl` files in the respective output directories:
- PHC: `assets/test_data/motion_retargeting/phc_retargeted/`
- Mink: `assets/test_data/motion_retargeting/mink_retargeted/`

**Configuration**: 
- PHC: Modify `phc_config.yaml`
- Mink: Modify `mink_config.yaml`

### 3. Result Visualization

Generate video outputs to validate retargeting quality:

```bash
cd {holomotion_path}
bash ./holomotion/scripts/motion_retargeting/run_motion_viz_mujoco.sh
```

**Output**: `{motion_name}.mp4` files in the retargeted data directories

**Configuration**: Modify `mujoco_viz_config.yaml` for visualization settings

## Supported Robots

Currently supported robot configurations:

- **Unitree G1**: 21DOF

## Output Format

Retargeted motions are saved in a standardized format containing:

```python
{
    "pose_aa": np.ndarray,              # Joint angles in axis-angle format
    "dof": np.ndarray,                  # Degrees of freedom positions
    "root_rot": np.ndarray,             # Root rotation (quaternion)
    "root_trans_offset": np.ndarray,    # Root translation
}
```

