#!/bin/bash

docker kill holomotion_orin_deploy
docker rm holomotion_orin_deploy
echo "Old holomotion_orin_deploy container removed !"

# Initialize variable as empty
holomotion_repo_path=""

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

sudo docker run -it \
  --name holomotion_orin_deploy \
  --runtime nvidia \
  --gpus all \
  --privileged \
  --network host \
  -e "ACCEPT_EULA=Y" \
  -v "$holomotion_repo_path:/home/unitree/holomotion" \
  -v "/usr/local/cuda-11.4/targets/aarch64-linux/lib:/cuda_base:ro" \
  -v "/usr/lib/aarch64-linux-gnu/libcudnn.so.8.6.0:/host_gpu/libcudnn.so.8.6.0:ro" \
  -v "/usr/lib/aarch64-linux-gnu/libcudnn_ops_infer.so.8.6.0:/host_gpu/libcudnn_ops_infer.so.8.6.0:ro" \
  -v "/usr/lib/aarch64-linux-gnu/libcudnn_cnn_infer.so.8.6.0:/host_gpu/libcudnn_cnn_infer.so.8.6.0:ro" \
  holomotion_orin_foxy_jp5.1_docker_humble_deploy:20260105 \
  bash -c "ln -sf /host_gpu/libcudnn.so.8.6.0 /host_gpu/libcudnn.so.8 && \
           ln -sf /host_gpu/libcudnn_ops_infer.so.8.6.0 /host_gpu/libcudnn_ops_infer.so.8 && \
           ln -sf /host_gpu/libcudnn_cnn_infer.so.8.6.0 /host_gpu/libcudnn_cnn_infer.so.8 && \
           source /root/miniconda3/bin/activate && conda activate holomotion_deploy && exec bash"