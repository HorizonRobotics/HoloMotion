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
├── HoloMotion_assets
├── GMR
├── smplx
├── joints2smpl
├── omomo_release
├── smpl_models
├── SMPLSim
├── unitree_ros
└── unitree_ros2
```

## Step 3: Create the Conda Environment

Create the conda environment named `holomotion_train` and `holomotion_deploy`:

```shell
conda env create -f environments/environment_train_isaaclab_cu118.yaml
# for newer GPUs like RTX 5090, use environment_train_isaaclab_cu128.yaml
conda env create -f environments/environment_deploy.yaml
```


Install smplx and GMR into the conda environment:

```shell
cd thirdparties

conda activate holomotion_train

pip install -e ./smplx

# use --no-deps to avoid pulling GMR's dependencies
pip install -e ./GMR --no-deps
```

## Step 4: Configure the Environment Variables

HoloMotion uses `train.env` and `deploy.env` files to export environment variables in the shell entry scripts. Please make sure the `Train_CONDA_PREFIX` and the `Deploy_CONDA_PREFIX` variables in `train.env` and `deploy.env` are correctly setup. You can manually source these files and check the output in the shell.

Take the `train.env` for example:

```shell
source train.env
```

These `.env` files will be sourced in the shell scripts (in `holomotion/scripts`) to correctly find and utilize your conda environments.
