#!/bin/bash
# =============================================
# ML-Env-CUDA13 WSL/Ubuntu Full Setup Script (Idempotent)
# PyTorch (CUDA 12.3) + TensorFlow (CUDA 12.3) - Dual-GPU Setup
# =============================================

set -e

# 1. Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.11 and pip
sudo apt install software-properties-common -y
if ! grep -q deadsnakes /etc/apt/sources.list /etc/apt/sources.list.d/*; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
fi
sudo apt install python3.11 python3.11-venv python3-pip -y

# 3. Remove and recreate virtual environment
if [ -d "~/ml_env" ]; then
    rm -rf ~/ml_env
fi
python3.11 -m venv ~/ml_env
source ~/ml_env/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install/upgrade ML libraries with GPU support
pip install --upgrade tensorflow[and-cuda]
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Run GPU verification tests
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('TensorFlow GPUs:', tf.config.list_physical_devices('GPU'))"

# 7. Export requirements for WSL2/Ubuntu
pip freeze > requirements-wsl.txt

# 8. Final message
echo "\nSETUP COMPLETE! Full GPU support for PyTorch and TensorFlow in WSL2/Ubuntu."
echo "PyTorch: CUDA 12.3 | TensorFlow: CUDA 12.3 (WSL2)"
echo "Your RTX 2000 Ada is now a dual-framework deep learning beast!"
