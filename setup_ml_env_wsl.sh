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

# 6. Run GPU verification tests and detect CUDA versions
python - <<'PY'
import sys
out = {}
try:
    import torch
    out['torch_version'] = torch.__version__
    out['torch_cuda_version'] = getattr(torch.version, 'cuda', None) or getattr(torch, 'version', None)
    out['torch_cuda_available'] = torch.cuda.is_available()
except Exception as e:
    out['torch_error'] = str(e)

try:
    import tensorflow as tf
    out['tf_version'] = tf.__version__
    try:
        build = tf.sysconfig.get_build_info()
        out['tf_cuda_version'] = build.get('cuda_version') or build.get('cuda_version_text') or None
        out['tf_cudnn_version'] = build.get('cudnn_version') or None
    except Exception:
        out['tf_cuda_version'] = None
    out['tf_gpus'] = tf.config.list_physical_devices('GPU')
except Exception as e:
    out['tf_error'] = str(e)

import json
print(json.dumps(out, indent=2))
PY

# 7. Also report system CUDA tool/runtime info if available
echo "\nSystem NVIDIA / CUDA info (nvidia-smi, nvcc)"
nvidia-smi || true
nvcc --version 2>/dev/null || echo "nvcc not on PATH or not installed in WSL"

# 8. Export requirements for WSL2/Ubuntu
pip freeze > requirements-wsl.txt

# 9. Final message (use detected info when possible)
echo "\nSETUP COMPLETE! Full GPU support for PyTorch and TensorFlow in WSL2/Ubuntu."
echo "Check 'requirements-wsl.txt' and the JSON verification output above for exact package and CUDA info."
echo "Your RTX 2000 Ada is now a dual-framework deep learning beast!"

# 10. Create basic test scripts if they don't exist (idempotent)
if [ ! -f "test_tensorflow.py" ]; then
cat > test_tensorflow.py <<'PY'
import json
import subprocess
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Detected: {len(gpus) > 0}")
for i, gpu in enumerate(gpus):
    try:
        name = gpu.name
    except Exception:
        name = repr(gpu)
    print(f"GPU {i}: {name}")

try:
    build = tf.sysconfig.get_build_info()
    cuda_build = build.get('cuda_version') or build.get('cuda_version_text') or None
    cudnn_build = build.get('cudnn_version') or None
    print("TensorFlow build info:")
    print(json.dumps({
        'tf_version': tf.__version__,
        'cuda_build': cuda_build,
        'cudnn_build': cudnn_build,
    }, indent=2))
except Exception as e:
    print("Could not retrieve TensorFlow build info:", e)

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        return out.strip()
    except Exception as e:
        return f"Error running {' '.join(cmd)}: {e}"

print('\nSystem NVIDIA / CUDA info (nvidia-smi, nvcc)')
print(run_cmd(['nvidia-smi']))
nvcc_out = run_cmd(['nvcc','--version'])
if 'Error running' in nvcc_out:
    print('nvcc not on PATH or not installed in WSL')
else:
    print(nvcc_out)
PY
fi

if [ ! -f "test_pytorch.py" ]; then
cat > test_pytorch.py <<'PY'
import json
import subprocess
import torch

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        return out.strip()
    except Exception as e:
        return f"Error running {' '.join(cmd)}: {e}"

print(f"PyTorch: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"GPU Detected: {cuda_available}")
if cuda_available:
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = repr(torch.cuda.current_device())
else:
    gpu_name = 'None'
print(f"GPU 0: {gpu_name}")

# Build info
cuda_build = None
try:
    cuda_build = getattr(torch.version, 'cuda', None) or torch.version.cuda
except Exception:
    cuda_build = None
try:
    cudnn_build = torch.backends.cudnn.version()
except Exception:
    cudnn_build = None

build_info = {
    'torch_version': torch.__version__,
    'cuda_build': cuda_build,
    'cudnn_build': cudnn_build,
    'build_info_sample': {
        'cuda_version': cuda_build,
        'cudnn_version': cudnn_build,
    }
}

print("\nPyTorch build info:")
print(json.dumps(build_info, indent=2))

print('\nSystem NVIDIA / CUDA info (nvidia-smi, nvcc)')
print(run_cmd(['nvidia-smi']))
nvcc_out = run_cmd(['nvcc', '--version'])
if 'Error running' in nvcc_out:
    print('nvcc not on PATH or not installed in WSL')
else:
    print(nvcc_out)
PY
fi
