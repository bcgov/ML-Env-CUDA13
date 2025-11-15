
#!/bin/bash
#!/bin/bash
# =============================================
# ML-Env-CUDA13 WSL/Ubuntu Full Setup Script (Idempotent)
# This script is safe to run multiple times. It supports --regen-tests-only
# which overwrites test files and exits without installing packages.

set -euo pipefail

# --------- Configuration / CLI flags ---------
# Default CUDA_TAG; may be overridden by env var or auto-detected from nvcc
CUDA_TAG=${CUDA_TAG:-cu126}
REGEN_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --regen-tests-only)
            REGEN_ONLY=true
            ;;
        *)
            ;;
    esac
done

# If nvcc exists, try to auto-detect a matching CUDA_TAG (nvcc 12.6 -> cu126)
if command -v nvcc >/dev/null 2>&1; then
    nvcc_ver=$(nvcc --version 2>/dev/null | sed -n "s/.*release \([0-9]\+\.[0-9]\+\).*/\1/p" | head -n1 || true)
    if [ -n "${nvcc_ver:-}" ]; then
        major=$(echo "$nvcc_ver" | cut -d. -f1)
        minor=$(echo "$nvcc_ver" | cut -d. -f2)
        # form cu<major><minor> (e.g., 12.6 -> cu126)
        CUDA_TAG_AUTO="cu${major}${minor}"
        CUDA_TAG=${CUDA_TAG_AUTO:-$CUDA_TAG}
    fi
fi

PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

regen_write_tests() {
    echo "Writing/overwriting GPU test templates (CUDA_TAG=$CUDA_TAG)"
    cat > test_torch_cuda.py <<'PY'
import torch
import sys
print('torch.__version__ =', torch.__version__)
print('cuda_available =', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count =', torch.cuda.device_count())
    try:
        print('cuda_device_name =', torch.cuda.get_device_name(0))
    except Exception:
        print('cuda_device_name = unknown')
    try:
        print('cudnn_version =', torch.backends.cudnn.version())
    except Exception:
        print('cudnn_version = unknown')
else:
    sys.exit(2)
PY

    cat > test_xformers.py <<'PY'
try:
    import xformers
    print('xformers import OK; version =', getattr(xformers, '__version__', 'unknown'))
except Exception as e:
    print('xformers import failed:', e)
    raise
PY

    cat > test_llama_cpp.py <<'PY'
try:
    import llama_cpp
    print('llama_cpp import OK')
except Exception as e:
    print('llama-cpp-python import failed:', e)
    raise
PY

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
}

if [ "$REGEN_ONLY" = true ]; then
    regen_write_tests
    echo "Wrote test_torch_cuda.py, test_xformers.py, test_llama_cpp.py, test_pytorch.py, test_tensorflow.py"
    exit 0
fi

# 1. Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.11 and pip
sudo apt install software-properties-common -y
if ! grep -q deadsnakes /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
fi
sudo apt install python3.11 python3.11-venv python3-pip -y

# 3. Remove and recreate virtual environment (use $HOME for portability)
if [ -d "$HOME/ml_env" ]; then
    echo "[INFO] Removing existing virtualenv at $HOME/ml_env"
    rm -rf "$HOME/ml_env"
fi
python3.11 -m venv "$HOME/ml_env"
# shellcheck source=/dev/null
source "$HOME/ml_env/bin/activate"

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install/upgrade ML libraries with GPU support (core packages)
# NOTE: there is no `and-cuda` extra for TensorFlow. Install plain tensorflow
# (or pin a specific compatible version like `tensorflow==2.20.0` if required).
pip install --upgrade tensorflow || true
# Install PyTorch wheels that match CUDA_TAG via the appropriate index
# (pin versions externally if you need deterministic builds).
pip install --upgrade torch torchvision torchaudio --index-url "$PYTORCH_INDEX" || true

# 6. Run GPU verification tests (core gate)
LOG_DIR="$(pwd)/ml_env_logs"
mkdir -p "$LOG_DIR"
echo "[INFO] Running core verification: test_torch_cuda.py"
if python test_torch_cuda.py > "$LOG_DIR/test_torch_cuda.log" 2>&1; then
    echo "[INFO] Torch verification passed. Logs: $LOG_DIR/test_torch_cuda.log"
    # Snapshot environment after successful core gate
    SNAPSHOT="$(pwd)/pinned-requirements-$(date +%Y%m%d%H%M).txt"
    pip freeze > "$SNAPSHOT"
    echo "[INFO] Created snapshot: $SNAPSHOT"
else
    echo "[ERROR] Torch verification failed; see $LOG_DIR/test_torch_cuda.log"
    exit 1
fi

# optional installs (non-fatal)
echo "[INFO] Running optional tests (non-fatal): xformers and llama-cpp-python"
python test_xformers.py > "$LOG_DIR/test_xformers.log" 2>&1 || echo "[WARN] xformers test failed; see $LOG_DIR/test_xformers.log"
python test_llama_cpp.py > "$LOG_DIR/test_llama_cpp.log" 2>&1 || echo "[WARN] llama-cpp-python test failed; see $LOG_DIR/test_llama_cpp.log"

# Optional GPU postinstall (non-fatal)
if [ -f requirements-gpu-postinstall.txt ]; then
  echo "[INFO] Installing optional GPU postinstall requirements (non-fatal)"
  pip install -r requirements-gpu-postinstall.txt > "$LOG_DIR/postinstall.log" 2>&1 || echo "[WARN] postinstall step failed; see $LOG_DIR/postinstall.log"
fi

# 7. Also report system CUDA tool/runtime info if available
echo "\nSystem NVIDIA / CUDA info (nvidia-smi, nvcc)"
nvidia-smi || true
nvcc --version 2>/dev/null || echo "nvcc not on PATH or not installed in WSL"

# 8. Export requirements for WSL2/Ubuntu (already snapshot above)
pip freeze > requirements-wsl.txt

echo "\nSETUP COMPLETE! Check $LOG_DIR for logs and the pinned snapshot created after verification."

