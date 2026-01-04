#!/bin/bash
# ====================================================================================
# ML-Env-CUDA13: Comprehensive WSL/Ubuntu Environment Setup (Idempotent)
# ====================================================================================
#
# PURPOSE:
#   Creates a Reproducible, Hybrid AI/ML Development Environment suitable for
#   Project Sanctuary and other advanced LLM workflows.
#
# ARCHITECTURE (Hybrid Dependency Management - ADR 001):
#   1. Foundation Layer (Dynamic):
#      - Installs hardware-specific binaries (PyTorch, TensorFlow, xformers).
#      - Adapts to flags (e.g., --cuda13 for Nightly, default for Stable).
#      - Managed via standard `pip install` to handle complex binary linkages.
#
#   2. Application Layer (Strict):
#      - Installs higher-level libraries (rich, dataset, fine-tuning stack).
#      - Managed via `pip-tools`.
#      - AUTOMATICALLY recompiles `requirements.txt` from `requirements.in`
#        on every run to ensure the lockfile matches the active Foundation Layer.
#
# FEATURES:
#   - Idempotent: Can be run multiple times to repair/update the environment.
#   - Auto-Verification: Runs a suite of tests (Torch, CUDA, xformers, Llama.cpp)
#     immediately after setup to guarantee operational status.
#   - Project Structure Awareness: Generates test scripts in `tests/` and logs
#     to `ml_env_logs/`.
#
# USAGE:
#   bash scripts/setup_ml_env_wsl.sh            # Standard (Stable / CUDA 12.x)
#   bash scripts/setup_ml_env_wsl.sh --cuda13   # Experimental (Nightly / CUDA 13.0)
#
# Running time: ~5 minutes
#
# ====================================================================================
# ML-Env-CUDA13: Comprehensive WSL/Ubuntu Environment Setup (Idempotent)
# ====================================================================================

# Record start time
SCRIPT_START_TIME=$(date +%s)

set -euo pipefail

# --------- Configuration / CLI flags ---------
# Defaults
CUDA_TAG="cu126"  # Default to 12.6 (Stable)
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
PIP_PRE_FLAG="" # Empty by default (stable)
REGEN_ONLY=false

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --cuda13            Install PyTorch Nightly with CUDA 13.0 support"
    echo "  --regen-tests-only  Regenerate verification scripts in tests/ without installing"
    echo "  --help              Show this message"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --cuda13)
            echo "[INFO] Switching to CUDA 13.0 (PyTorch Nightly)"
            CUDA_TAG="cu130"
            PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"
            PIP_PRE_FLAG="--pre"
            ;;
        --regen-tests-only)
            REGEN_ONLY=true
            ;;
        --help)
            usage
            ;;
        *)
            echo "[WARN] Unknown argument: $arg"
            ;;
    esac
done

# --------- Test Generation ---------
regen_write_tests() {
    TEST_DIR="$(pwd)/tests"
    mkdir -p "$TEST_DIR"
    echo "[INFO] Writing GPU test templates to $TEST_DIR (CUDA_TAG=$CUDA_TAG)"

    # test_torch_cuda.py
    cat > "$TEST_DIR/test_torch_cuda.py" <<'PY'
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

    # test_xformers.py
    cat > "$TEST_DIR/test_xformers.py" <<'PY'
try:
    import xformers
    print('xformers import OK; version =', getattr(xformers, '__version__', 'unknown'))
except Exception as e:
    print('xformers import failed:', e)
    raise
PY

    # test_llama_cpp.py
    cat > "$TEST_DIR/test_llama_cpp.py" <<'PY'
try:
    import llama_cpp
    print('llama_cpp import OK')
    from llama_cpp import Llama
    print('Llama class available')
except Exception as e:
    print('llama-cpp-python import failed:', e)
    raise
PY

    # test_pytorch.py (General)
    cat > "$TEST_DIR/test_pytorch.py" <<'PY'
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

    # test_tensorflow.py
    cat > "$TEST_DIR/test_tensorflow.py" <<'PY'
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
    echo "Wrote verification scripts to tests/"
    exit 0
fi

# =============================================
# INSTALLATION
# =============================================

# 1. Update and upgrade system packages
echo "[INFO] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.11 and pip
echo "[INFO] Installing Python 3.11..."
sudo apt install software-properties-common -y
if ! grep -q deadsnakes /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
fi
sudo apt install python3.11 python3.11-venv python3-pip python3.11-dev -y

# 3. Environment Creation
# Use $HOME/ml_env for portability
VENV_PATH="$HOME/ml_env"
if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Removing existing virtualenv at $VENV_PATH"
    rm -rf "$VENV_PATH"
fi
echo "[INFO] Creating venv at $VENV_PATH"
python3.11 -m venv "$VENV_PATH"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

# 4. Upgrade pip
pip install --upgrade pip

# =============================================
# LAYER 0: PURGE (Prevent Symbol Mismatches)
# =============================================
echo "[INFO] Purging stale NVIDIA/NCCL packages to prevent symbol mismatches..."
# Uninstall any existing cu12 or cu13 nvidia libraries that might conflict with the new build
# This resolves errors like "undefined symbol: ncclDevCommDestroy"
pip freeze | grep -E "nvidia-.*-cu1|nccl" | cut -d @ -f 1 | xargs -r pip uninstall -y || true

# =============================================
# LAYER 1: FOUNDATION (Dynamic / Exception)
# =============================================
echo "[INFO] Installing Foundation Layer (Torch/TF) - Index: $PYTORCH_INDEX"

# Install TensorFlow (Always stable for now, CPU-only logic handled by pip if incompatible)
# Note: TF Nightly/CUDA 13 support is experimental, sticking to standard TF for now.
pip install --upgrade tensorflow || true

# Install PyTorch (Dynamic Tag)
# Uses $PIP_PRE_FLAG (e.g., --pre) if CUDA 13 is requested
pip install $PIP_PRE_FLAG --upgrade torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Install xformers (Hardware-sensitive, Foundation Layer)
# NOTE: We skip xformers on CUDA 13 (Nightly) because PIP will aggressively 
# downgrade Torch 2.11 to satisfy the missing xformers cu130 wheels.
if [ "$CUDA_TAG" = "cu130" ]; then
    echo "[INFO] Skipping xformers for CUDA 13 (prevents dependency conflicts)"
else
    echo "[INFO] Installing xformers..."
    pip install xformers || echo "[WARN] xformers installation failed"
fi

# =============================================
# LAYER 2: APPLICATION (Hybrid / Strict)
# =============================================
echo "[INFO] Installing Application Layer (via pip-tools)"

pip install pip-tools

# Requirement: Core Application
if [ -f requirements.in ]; then
    echo "[INFO] Compiling requirements.txt from requirements.in..."
    # Force fresh compilation to match the current Foundation Layer (e.g. CUDA 13 vs 12)
    # This prevents stale lockfiles from pinning inconsistent versions (like torch 2.9 vs 2.11)
    # We explicitly pass the index URL and pre-release flag so pip-compile sees the Nightly wheels
    pip-compile requirements.in --upgrade --extra-index-url "$PYTORCH_INDEX" $PIP_PRE_FLAG
fi
echo "[INFO] Installing from requirements.txt..."
pip install -r requirements.txt

# Requirement: Development Tools
if [ -f requirements-dev.in ]; then
    if [ ! -f requirements-dev.txt ]; then
        echo "[INFO] requirements-dev.txt not found. Compiling..."
        pip-compile requirements-dev.in --output-file requirements-dev.txt
    fi
    echo "[INFO] Installing from requirements-dev.txt..."
    pip install -r requirements-dev.txt
fi

# =============================================
# VERIFICATION
# =============================================

# Regenerate test files to ensure they match current logic
regen_write_tests

LOG_DIR="$(pwd)/ml_env_logs"
mkdir -p "$LOG_DIR"

echo "[INFO] Running core verification: tests/test_torch_cuda.py"
if python tests/test_torch_cuda.py > "$LOG_DIR/test_torch_cuda.log" 2>&1; then
    echo "[INFO] Torch verification passed. Logs: $LOG_DIR/test_torch_cuda.log"
    
    # Snapshot (Optional now, as we have requirements.txt, but good for debugging foundation)
    SNAPSHOT="$(pwd)/archive/pinned-requirements-$(date +%Y%m%d%H%M).txt"
    mkdir -p archive
    pip freeze > "$SNAPSHOT"
    echo "[INFO] Created full freeze snapshot: $SNAPSHOT"
else
    echo "[ERROR] Torch verification failed; see $LOG_DIR/test_torch_cuda.log"
    echo "Tail of log:"
    tail -n 5 "$LOG_DIR/test_torch_cuda.log"
    exit 1
fi

# Optional Tests
echo "[INFO] Running optional tests (non-fatal)..."
python tests/test_xformers.py > "$LOG_DIR/test_xformers.log" 2>&1 || echo "[WARN] xformers check failed"
python tests/test_llama_cpp.py > "$LOG_DIR/test_llama_cpp.log" 2>&1 || echo "[WARN] llama-cpp check failed"

# Report
SCRIPT_END_TIME=$(date +%s)
DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================="
echo "SETUP COMPLETE!"
echo "Environment: $VENV_PATH"
echo "CUDA Target: $CUDA_TAG"
echo "Log Dir:     $LOG_DIR"
echo "Total Time:  ${MINUTES}m ${SECONDS}s"
echo "============================================="
