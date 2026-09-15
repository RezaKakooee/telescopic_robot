#!/usr/bin/env bash
set -euo pipefail

export PIP_CACHE_DIR="/home/storage_group/pip_cache"
export TMPDIR="/home/storage_group/pip_cache/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

ENV_DIR="/home/storage_group/envs/lerobot"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

echo "=== 1. Installing PyTorch with CUDA 12.6 in lerobot environment ==="
"${PIP}" install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "=== 2. Installing LeRobot and VLA dependencies ==="
"${PIP}" install "lerobot[smolvla]" "opencv-python-headless" "diffusers" "accelerate" "datasets"

echo "=== 3. Installing Simulation & Environment utilities ==="
"${PIP}" install "mujoco==3.8.1" gymnasium omegaconf imageio imageio-ffmpeg scipy   # 3.8.1: the skills are calibrated on it; 3.13 changes the contacts

echo "=== 4. Verifying Installation ==="
"${PYTHON}" -c "
import torch
import lerobot
import mujoco
print('=' * 50)
print('LeRobot Environment Verification Successful!')
print(f'PyTorch version: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'LeRobot version: {lerobot.__version__}')
print(f'MuJoCo version:  {mujoco.__version__}')
print('=' * 50)
"
