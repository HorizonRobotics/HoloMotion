#!/usr/bin/env bash
set -euo pipefail

# One-click setup for the HoloMotion teleoperation environment.
#
# Runtime path:
#   XRoboToolkit body_poses[24, 7] -> HoloRetarget -> reference_qpos[36] ZMQ
#
# Usage:
#   bash setup_holomotion_teleop_x86_ubuntu2204.sh
#
# Optional env vars:
#   ENV_NAME=holomotion_teleop
#   PYTHON_VERSION=3.12
#   INSTALL_APT_DEPS=0
#   THIRD_PARTY_DIR=/path/to/third_party
#   XRT_PYBIND_REPO_DIR=/path/to/XRoboToolkit-PC-Service-Pybind

ENV_NAME="${ENV_NAME:-holomotion_teleop}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$REPO_ROOT/third_party}"

XRT_PYBIND_REPO_URL="${XRT_PYBIND_REPO_URL:-https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git}"
XRT_PC_SERVICE_REPO_URL="${XRT_PC_SERVICE_REPO_URL:-https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git}"
XRT_PYBIND_REPO_DIR="${XRT_PYBIND_REPO_DIR:-$THIRD_PARTY_DIR/XRoboToolkit-PC-Service-Pybind}"

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

error() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_command() {
  local cmd="$1"
  local hint="${2:-}"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    if [[ -n "$hint" ]]; then
      error "$cmd not found. $hint"
    else
      error "$cmd not found."
    fi
  fi
}

run_conda_relaxed() {
  set +u
  "$@"
  local status=$?
  set -u
  return $status
}

show_env_summary() {
  info "repo root: $REPO_ROOT"
  info "teleop dir: $SCRIPT_DIR"
  info "env name: $ENV_NAME"
  info "python version: $PYTHON_VERSION"
  info "install apt deps: $INSTALL_APT_DEPS"
  info "third party dir: $THIRD_PARTY_DIR"
  info "xrt pybind dir: $XRT_PYBIND_REPO_DIR"
}

check_platform() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    error "This setup script currently supports Linux only."
  fi

  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    info "detected OS: ${PRETTY_NAME:-unknown}"
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
      warn "This script is primarily tested on Ubuntu 22.04. Continuing anyway."
    fi
  fi
}

install_apt_deps_if_needed() {
  case "$INSTALL_APT_DEPS" in
    0|false|False|FALSE|no|NO)
      info "Skipping apt dependency installation because INSTALL_APT_DEPS=$INSTALL_APT_DEPS"
      return
      ;;
    1|true|True|TRUE|yes|YES)
      ;;
    *)
      error "Unsupported INSTALL_APT_DEPS value: $INSTALL_APT_DEPS (expected 1 or 0)"
      ;;
  esac

  require_command sudo "Install sudo or run the equivalent apt commands manually."
  require_command apt-get "This script needs apt-get to install build tools."
  sudo apt-get update
  sudo apt-get install -y build-essential git cmake
}

setup_conda_env() {
  require_command conda "Please install Miniconda or Anaconda first."
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
    info "Creating conda env: $ENV_NAME"
    run_conda_relaxed conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  else
    info "Conda env already exists: $ENV_NAME"
  fi

  run_conda_relaxed conda activate "$ENV_NAME"
}

clone_repo_if_missing() {
  local repo_dir="$1"
  local repo_url="$2"
  local repo_name="$3"

  if [[ ! -d "$repo_dir/.git" ]]; then
    info "Cloning $repo_name"
    mkdir -p "$(dirname "$repo_dir")"
    git clone "$repo_url" "$repo_dir"
  else
    info "Using existing $repo_name checkout at $repo_dir"
  fi
}

ensure_holoretarget_assets() {
  local asset_root="$REPO_ROOT/holoretarget/assets"
  local required_config="$asset_root/target_configs/smplx_to_g1.json"
  local required_mjcf="$asset_root/unitree_g1/g1_mocap_29dof.xml"

  if [[ -f "$required_config" && -f "$required_mjcf" ]]; then
    info "HoloRetarget assets are ready: $asset_root"
    return
  fi

  [[ -f "$required_config" ]] || error "Missing required HoloRetarget config: $required_config"
  [[ -f "$required_mjcf" ]] || error "Missing required HoloRetarget MJCF: $required_mjcf"
}

build_xrt_python_sdk() {
  clone_repo_if_missing "$XRT_PYBIND_REPO_DIR" "$XRT_PYBIND_REPO_URL" "XRoboToolkit pybind repository"

  pushd "$XRT_PYBIND_REPO_DIR" >/dev/null

  mkdir -p tmp
  clone_repo_if_missing "tmp/XRoboToolkit-PC-Service" "$XRT_PC_SERVICE_REPO_URL" "XRoboToolkit PC Service source"

  info "Building PXREARobotSDK"
  pushd tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK >/dev/null
  bash build.sh
  popd >/dev/null

  mkdir -p lib include
  cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
  cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
  cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/

  info "Installing pybind11 into conda env"
  run_conda_relaxed conda install -y -c conda-forge pybind11

  info "Installing xrobotoolkit_sdk"
  python -m pip uninstall -y xrobotoolkit_sdk || true
  python setup.py install

  popd >/dev/null
}

install_runtime_python_deps() {
  info "Upgrading pip toolchain"
  python -m pip install --upgrade pip setuptools wheel

  info "Installing HoloRetarget runtime dependencies"
  python -m pip install --upgrade numpy pyzmq typing_extensions mujoco
  python -m pip install --upgrade "newton==1.0.0"
  python -m pip install --upgrade --extra-index-url https://pypi.nvidia.com "warp-lang==1.12.0"

  info "Installing repository package in editable mode"
  python -m pip install -e "$REPO_ROOT"
}

print_next_steps() {
  echo
  info "Environment setup complete"
  echo
  info "Manual prerequisite:"
  echo "  Install XRoboToolkit PC Service manually from the Ubuntu 22.04 .deb package."
  echo "  Launch xrobotoolkit-pc-service before teleoperation."
  echo
  info "Activate with:"
  echo "  conda activate $ENV_NAME"
  echo
  info "Example command:"
  echo "  python \"$SCRIPT_DIR/holomotion_teleop_node.py\" \\"
  echo "    --robot-zmq-uri tcp://*:6001 \\"
  echo "    --robot-zmq-mode bind \\"
  echo "    --hz 50 \\"
  echo "    --timing-log-every 250"
}

main() {
  check_platform
  show_env_summary
  install_apt_deps_if_needed
  setup_conda_env
  ensure_holoretarget_assets
  build_xrt_python_sdk
  install_runtime_python_deps
  print_next_steps
}

main "$@"
