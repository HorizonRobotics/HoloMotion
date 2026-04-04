#!/usr/bin/env bash
set -euo pipefail

# One-click setup script for the holomotion teleoperation environment.
#
# This script automates the manually verified workflow:
# 1. create/activate conda env
# 2. clone/install GMR
# 3. clone/build/install XRoboToolkit pybind SDK
# 4. clone/install SMPLSim
# 5. install runtime Python dependencies
#
# Usage:
#   bash setup_gmr_holomotion_teleop_ubuntu2204.sh
#
# Optional env vars:
#   ENV_NAME=holomotion_teleop
#   PYTHON_VERSION=3.10
#   INSTALL_APT_DEPS=0             # default disabled; set to 1 only if you need apt
#   THIRD_PARTY_DIR=/path/to/third_party
#   GMR_SOURCE_DIR=/path/to/GMR
#   SMPLSIM_SOURCE_DIR=/path/to/SMPLSim
#   XRT_PYBIND_REPO_DIR=/path/to/XRoboToolkit-PC-Service-Pybind

ENV_NAME="${ENV_NAME:-holomotion_teleop}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$PROJECT_ROOT/third_party}"

GMR_REPO_URL="${GMR_REPO_URL:-https://github.com/YanjieZe/GMR.git}"
SMPLSIM_REPO_URL="${SMPLSIM_REPO_URL:-https://github.com/ZhengyiLuo/SMPLSim.git}"
XRT_PYBIND_REPO_URL="${XRT_PYBIND_REPO_URL:-https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git}"
XRT_PC_SERVICE_REPO_URL="${XRT_PC_SERVICE_REPO_URL:-https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git}"

GMR_SOURCE_DIR="${GMR_SOURCE_DIR:-$THIRD_PARTY_DIR/GMR}"
SMPLSIM_SOURCE_DIR="${SMPLSIM_SOURCE_DIR:-$THIRD_PARTY_DIR/SMPLSim}"
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
  # Some conda activation/deactivation hooks are not compatible with `set -u`
  # and may reference unset variables such as SETVARS_CALL.
  set +u
  "$@"
  local status=$?
  set -u
  return $status
}

show_env_summary() {
  info "project root: $PROJECT_ROOT"
  info "env name: $ENV_NAME"
  info "python version: $PYTHON_VERSION"
  info "install apt deps: $INSTALL_APT_DEPS"
  info "third party dir: $THIRD_PARTY_DIR"
  info "gmr source dir: $GMR_SOURCE_DIR"
  info "smplsim source dir: $SMPLSIM_SOURCE_DIR"
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

apt_deps_missing() {
  local missing=0
  command -v gcc >/dev/null 2>&1 || missing=1
  command -v g++ >/dev/null 2>&1 || missing=1
  command -v make >/dev/null 2>&1 || missing=1
  command -v git >/dev/null 2>&1 || missing=1
  command -v cmake >/dev/null 2>&1 || missing=1
  return "$missing"
}

install_apt_deps_if_needed() {
  case "$INSTALL_APT_DEPS" in
    0|false|False|FALSE|no|NO)
      info "Skipping apt dependency installation because INSTALL_APT_DEPS=$INSTALL_APT_DEPS"
      info "This matches the manually verified workflow and avoids unrelated apt source failures"
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

  info "Installing apt packages needed for build"
  if ! sudo apt-get update; then
    error "apt-get update failed. Common causes: broken apt sources, third-party repository timeouts, or proxy/network issues."
  fi

  if ! sudo apt-get install -y build-essential git cmake; then
    cat >&2 <<'EOF'
[ERROR] apt package installation failed.

Try one of the following:
  1. sudo apt --fix-broken install
  2. disable broken third-party apt repositories temporarily
  3. rerun with INSTALL_APT_DEPS=0 if gcc/g++/make/git/cmake already exist
EOF
    exit 1
  fi
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

install_gmr() {
  clone_repo_if_missing "$GMR_SOURCE_DIR" "$GMR_REPO_URL" "GMR"
  info "Installing GMR in editable mode"
  python -m pip install -e "$GMR_SOURCE_DIR"
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

  info "Reinstalling xrobotoolkit_sdk"
  python -m pip uninstall -y xrobotoolkit_sdk || true
  python setup.py install

  popd >/dev/null
}

install_smplsim() {
  clone_repo_if_missing "$SMPLSIM_SOURCE_DIR" "$SMPLSIM_REPO_URL" "SMPLSim"
  info "Installing SMPLSim in editable mode"
  python -m pip install -e "$SMPLSIM_SOURCE_DIR"
}

install_runtime_python_deps() {
  info "Upgrading pip toolchain"
  python -m pip install --upgrade pip setuptools wheel

  info "Installing runtime Python packages"
  python -m pip install pyzmq
  python -m pip install open3d
}

install_compat_python_deps() {
  info "Installing compatibility packages"
  python -m pip install chumpy
  info "Pinning numpy for chumpy compatibility"
  python -m pip install --upgrade "numpy==1.23.5"
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
  echo "  python \"$PROJECT_ROOT/holomotion_teleop_node.py\" \\"
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
  install_gmr
  build_xrt_python_sdk
  install_runtime_python_deps
  install_smplsim
  install_compat_python_deps
  print_next_steps
}

main "$@"

