#!/usr/bin/env bash
# setup_instance.sh — runs on the remote Nebius instance to install dependencies.
# Called by run_on_nebius.sh over SSH. Do not run directly.
set -euo pipefail

REMOTE_DIR="${1:?Usage: setup_instance.sh <remote_dir>}"
cd "$REMOTE_DIR"

echo "==> Checking CUDA availability"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==> Checking Python"
python3 --version

echo "==> Creating virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --quiet --upgrade pip

echo "==> Installing PyTorch with CUDA support"
# Install torch separately with the CUDA wheel index to ensure GPU support.
# The version must match requirements.txt. Adjust cu124 if your instance has a
# different CUDA toolkit version (check with: nvcc --version or nvidia-smi).
pip install --quiet "torch==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing remaining requirements"
# Install everything except torch (already installed above).
grep -v "^torch==" requirements.txt | pip install --quiet -r /dev/stdin

echo "==> Setup complete"
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
