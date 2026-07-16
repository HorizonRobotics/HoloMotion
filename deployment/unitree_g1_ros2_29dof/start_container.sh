#!/bin/bash

set -euo pipefail

BASE_IMAGE="horizonrobotics/holomotion:orin_foxy_jp5.1_humble_deploy_zmq_full_20260509"
IMAGE_NAME="${HOLOMOTION_DEPLOY_IMAGE:-$BASE_IMAGE}"
CONTAINER_NAME="${HOLOMOTION_CONTAINER_NAME:-holomotion_orin_deploy}"
PIP_CACHE_DIR="${HOLOMOTION_PIP_CACHE_DIR:-$HOME/.cache/holomotion-pip}"
MODEL_ROOT="${HOLOMOTION_MODEL_ROOT:-}"

docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Old holomotion_orin_deploy container removed !"

holomotion_repo_path="${1:-}"

# Loop until the user provides a non-empty string
while [[ -z "$holomotion_repo_path" ]]; do
  read -p "Please enter the holomotion local repository path: " holomotion_repo_path
  
  if [[ -z "$holomotion_repo_path" ]]; then
    echo "Input cannot be empty."
  fi
done

# Validate the directory exists before running Docker
if [ ! -d "$holomotion_repo_path" ]; then
    echo "Error: Directory '$holomotion_repo_path' does not exist."
    exit 1
fi

echo "Mounting path: $holomotion_repo_path"
echo "Using image: $IMAGE_NAME"
mkdir -p "$PIP_CACHE_DIR"

model_mount_args=()
if [[ -n "$MODEL_ROOT" ]]; then
  if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "Error: Model root '$MODEL_ROOT' does not exist."
    exit 1
  fi
  MODEL_ROOT="$(cd "$MODEL_ROOT" && pwd -P)"
  echo "Mounting model root (read-only): $MODEL_ROOT"
  model_mount_args=(-v "$MODEL_ROOT:$MODEL_ROOT:ro")
fi

sudo docker run -it \
  --name "$CONTAINER_NAME" \
  --runtime nvidia \
  --gpus all \
  --privileged \
  --network host \
  -e "ACCEPT_EULA=Y" \
  -v "$holomotion_repo_path:/home/unitree/holomotion" \
  -v "$PIP_CACHE_DIR:/root/.cache/pip" \
  "${model_mount_args[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE_NAME" \
  -c "source /root/miniconda3/bin/activate && conda activate holomotion_deploy && ${HOLOMOTION_CONTAINER_COMMAND:-exec bash}"
