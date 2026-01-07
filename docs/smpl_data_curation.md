# Dataset Preparation Guide

This guide describes the workflow and setup for preparing datasets to train the motion tracking model.
We use **AMASS-compatible SMPL-format motion capture data** as the training input.

---

## Overview

The dataset preparation pipeline has the following steps:

1. **Download datasets**
   - To train with diverse and rich motion data, you first need to collect raw motion capture datasets from various sources.
   - Then place all downloaded datasets under the data/raw_datasets directory in their original structure.
2. **Convert datasets to AMASS format**
   - To ensure that all motion data is compatible with the AMASS-style .npz format used by the training pipeline, you need to convert the raw datasets.
   - Then run the conversion script to generate .npz files under data/amass_compatible_datasets/.
3. **Filter datasets**
   - To improve data quality by removing abnormal, noisy, or unwanted motion samples, you can optionally run the filtering step.
   - Then run the filtering script to generate filtered .yaml files under holomotion/config/data_curation/.
4. **Visualize Prepared Data**
   - Use the included visualization utility to preview and inspect the generated AMASS-compatible `.npz` motion files.
   - Quickly check for anomalies or errors before training.
5. **Generate Motion from Monocular Video**
   - You can also generate SMPL-format motion capture files **directly from monocular RGB videos** using GVHMR.
   - This allows you to create training data or test the model with real-world video footage.
   - Pipeline are given follow.

### Directory Structure After Full Setup

```
data/
├── raw_datasets/
│   ├── humanact12/
│   ├── OMOMO/
│   ├── MotionX/
│   └── ZJU_Mocap/
├── amass_compatible_datasets/
│   ├── amass/
│   │   ├── ACCAD/
│   │   ├── BioMotionLab_NTroje/
│   │   ├── ...
│   ├── humanact12/
│   ├── OMOMO/
│   ├── MotionX/
│   └── ZJU_Mocap/
├── dataset_labels/
│   ├── humanact12.jsonl
│   ├── OMOMO.jsonl
│   ├── MotionX.jsonl
│   ├── ZJU_Mocap.jsonl
│   ├── amass.jsonl
```

---

## Step-by-Step Instructions

### 1. Download Datasets

Download and extract the datasets into the `data/` folder as follows:

- `data/amass_compatible_datasets/amass/` (required)
  - [AMASS dataset](https://amass.is.tue.mpg.de/download.php) — choose **SMPL-X G** format.
- `data/raw_datasets/humanact12/` (optional)
  - [HumanAct12](https://github.com/EricGuo5513/action-to-motion?tab=readme-ov-file)
- `data/raw_datasets/OMOMO/` (optional)
  - [OMOMO dataset](https://github.com/lijiaman/omomo_release?tab=readme-ov-file)
- `data/raw_datasets/MotionX/` (optional)
  - [MotionX dataset](https://github.com/IDEA-Research/Motion-X)
- `data/raw_datasets/ZJU_Mocap/` (optional)
  - [EasyMocap](https://github.com/zju3dv/EasyMocap)

---

### 2. Convert Optional Datasets to AMASS Format (optional)

Skip this step if you only use amass dataset.

Step:

1. Initialize Submodules:
    Some datasets require external repositories or models for proper conversion. These are included as **submodules** under:
    ```
    thirdparties/
    ```
    Initialize them with:
    ```bash
    git submodule update --init --recursive
    ```
    If you need to modify or update these submodules, refer to their individual `README` files.


2. Download SMPL Model:
    - Download `SMPL_NEUTRAL.npz` from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php).
    - Place the file into:
    ```
    ./assets/smpl/
    ```

3. Modify `thirdparties/joints2smpl/src/customloss.py`:
    Before running the pipeline, make sure to modify the `body_fitting_loss_3d` function in `thirdparties/joints2smpl/src/customloss.py` to include the following change:
    ```python
    joint3d_loss = (joint_loss_weight ** 2) * joint3d_loss_part.sum(dim=-1)
    ```

4. Modify `thirdparties/joints2smpl/src/smplify.py`:
    Next, ensure the following modification in the `__call__` function of `SMPLify3D` inside `thirdparties/joints2smpl/src/smplify.py`:
    ```python
    init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).unsqueeze(1).detach()
    ```

5. Start Conversion:
    Run the provided script to convert all available datasets to AMASS `.npz` files:

    ```bash
    bash holomotion/scripts/data_curation/convert_to_amass.sh
    ```

    This script reads from `data/{dataset}/` and writes to `data/amass_compatible_datasets/{dataset}/`.

---

### 3. Filter Datasets (optional)

Skip this step if you prefer to use all available data for training.

#### Why filter?

The raw datasets may contain motions that are irrelevant, undesirable, or of poor quality for training. This step helps improve the overall quality and consistency of your dataset.

#### Filtering criteria

The filtering process excludes samples based on the following rules:

- **Upstairs/Downstairs motion:**  
  Paths containing keywords like stairs, staircase, upstairs, downstairs, or motions with large upward/downward Z translation and velocity are excluded.

- **Sitting motion:**  
  Sitting motion: Paths containing sitting/Sitting keywords or frames that match a reference sitting pose are excluded.

- **Known abnormal datasets:**  
  Known abnormal datasets: Samples from datasets like aist is excluded.

- **Unrealistic velocity:**  
  Unrealistic velocity: Motions where the mean velocity exceeds a threshold (default: 100.0) are excluded.

#### How to run

You can use the `-l` option to specify which datasets to filter (space-separated list).  
Run the filtering script to identify and exclude abnormal or unwanted samples:

```bash
bash holomotion/scripts/data_curation/filter_smpl_data.sh -l "amass humanact12 OMOMO MotionX ZJU_Mocap"
```

The output `.yaml` files will be placed in `holomotion/config/data_curation/`.

---

## Notes

- Paths are relative to the project root.
- The AMASS dataset must be manually requested from their website.
- Dataset conversion and filtering may take time depending on your hardware.

---

This guide assumes that you only need the basic configuration to run the complete pipeline. For further customization, refer to the relevant scripts in the repository and optional steps in the documentation.

## Video2SMPL Instructions
### 1. Environment setup
Create a new conda environment named 'gvhmr' following the official installation guide  
[[GVHMR setup doc](../thirdparties/GVHMR/docs/INSTALL.md)]
> Reminder: If you encounter 'hmr4d' missing module errors during runtime, install the following dependency separately in gvhmr env
```bash
pip install hmr4d
```

Rename the SMPL model files as follows  
basicmodel_{GENDER}_lbs_*.pkl  →  SMPL_{GENDER}.pkl  
Place the SMPL and SMPL-X model files into the directory structure below

```
thirdparties/GVHMR/inputs/checkpoints/
├── body_models/smplx
│   └── SMPLX_{GENDER}.npz
└── body_models/smpl
    └── SMPL_{GENDER}.pkl
```

### 2. Video to SMPL motion data
Confirm that all input videos have a frame rate of 30 FPS to avoid motion acceleration or deceleration.
```bash
bash ./holomotion/scripts/data_curation/video2SMPL_gvhmr.sh
```
> Reminder: Set the directory in the .sh file to an absolute path.

### 3. Convert SMPL data to SMPLX
Motion data from GVHMR are SMPL format.
Use ./thirdparties/GMR/scripts/smpl_to_smplx.py converting format to SMPLX for retargeting.