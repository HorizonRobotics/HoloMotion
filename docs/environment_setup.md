# Environment Setup

## Step 1: Setup Conda

This project uses conda to manage Python environments. We recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-installer).

**For users in China:** Configure the conda mirror following [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) for faster downloads.

## Step 2: Setup Third-party Dependencies

### 2.1 Download SMPL/SMPLX Models

We use SMPL/SMPLX models to retarget mocap data into robot motion data. Register your account and download the models from:

- [SMPL](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip)
- [SMPLX](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip)

Place both zip files (`SMPL_python_v.1.1.0.zip` and `models_smplx_v1_1.zip`) in the `thirdparties/` folder, then extract:

```shell
mkdir thirdparties/smpl_models
unzip thirdparties/SMPL_python_v.1.1.0.zip -d thirdparties/smpl_models/
unzip thirdparties/models_smplx_v1_1.zip -d thirdparties/smpl_models/
```

The resulting file structure for smpl models would be:

```shell
thirdparties/
├── smpl_models
   ├── models
   └── SMPL_python_v.1.1.0
```

### 2.2 Pull Submodules

After cloning this repository, run the following command to get all submodule dependencies:

```shell
git submodule update --init --recursive
```

### 2.3 Create Asset Symlinks

This project uses symbolic links to connect robot and SMPL assets from submodules to the main `assets` directory. Symlinks are created automatically when you clone the repository.

### 2.4 Verify Third-party File Structure

After completing the above steps, your file structure should look like this:

```shell
thirdparties/
├── GVHMR
├── HoloMotion_assets
├── SMPLSim
├── cyclonedds
├── joints2smpl
├── omomo_release
├── smpl_models
├── smplx
├── unitree_ros
├── unitree_ros2
├── unitree_sdk2
└── unitree_sdk2_python
```

## Step 3: Create the Conda Environment

Create the `holomotion_train` Conda environment. The files under `environments/` are the supported dependency installation entry points for v1.4.0. The project does not support installing runtime dependencies directly from `pyproject.toml`; its editable installation only registers the repository source after the Conda environment has been created.

Robot-side deployment uses the Docker workflow documented in [Real-World Deployment](./realworld_deployment.md).

```shell
conda env create -f environments/environment_train_isaaclab_cu118.yaml

# For newer GPUs like RTX 5090, create the cu128 environment first, then
# apply the project-validated Torch override as a separate pip operation.
conda env create -f environments/environment_train_isaaclab_cu128.yaml
conda run -n holomotion_train \
  python -m pip install -r environments/requirements_torch_cu128.txt
```

The separate override is required because Isaac Sim and the project-validated cu128 stack specify different exact Torch versions; requesting both in one pip operation would fail dependency resolution. The v1.4.0 cu128 environment uses Torch 2.9.1, torchvision 0.24.1, and torchaudio 2.9.1. Isaac Sim 5.0 declares exact dependencies on Torch 2.7.0, torchvision 0.22.0, and torchaudio 2.7.0, so `pip check` reports those three known version conflicts. Do not treat additional dependency conflicts as expected.


Install `smplx` into the conda environment:

```shell
cd thirdparties

conda activate holomotion_train

pip install -e ./smplx
```

## Step 4: Configure the Training Environment Variables

HoloMotion uses `train.env` to export the training environment variables used by shell entry scripts. Source it to verify that `Train_CONDA_PREFIX` points to the Conda environment created above:

```shell
source train.env
```

The scripts under `holomotion/scripts` source this file to locate the training environment. Do not create or configure a host-side deployment Conda environment. Robot deployment is Docker-only; the deployment image provides its own environment as described in [Real-World Deployment](./realworld_deployment.md).
