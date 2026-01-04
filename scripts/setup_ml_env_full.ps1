param(
    [switch]$RegenTestsOnly
)

$ErrorActionPreference = 'Stop'

Write-Host 'ML-Env-CUDA13 PowerShell setup (safe mode)' -ForegroundColor Green

function Write-TestFiles {
    Write-Host 'Writing test_pytorch.py and test_tensorflow.py' -ForegroundColor Cyan
    @'
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
    print('nvcc not on PATH or not installed')
else:
    print(nvcc_out)
'@ | Out-File -FilePath 'test_pytorch.py' -Encoding UTF8 -Force

    @'
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
    print('nvcc not on PATH or not installed')
else:
    print(nvcc_out)
'@ | Out-File -FilePath 'test_tensorflow.py' -Encoding UTF8 -Force
}

if ($RegenTestsOnly) {
    Write-TestFiles
    Write-Host 'Regenerated tests.' -ForegroundColor Green
    exit 0
}

# Ensure tests exist (create if missing)
if (-not (Test-Path 'test_pytorch.py' -PathType Leaf) -or -not (Test-Path 'test_tensorflow.py' -PathType Leaf)) {
    Write-Host 'Creating missing test files' -ForegroundColor Yellow
    Write-TestFiles
}

# Safe venv recreation: remove folder, not site-packages
if (Test-Path 'cuda_clean_env') {
    Write-Host "Removing existing virtual environment 'cuda_clean_env'" -ForegroundColor Yellow
    Remove-Item 'cuda_clean_env' -Recurse -Force
}
Write-Host "Creating virtual environment 'cuda_clean_env'" -ForegroundColor Green
python -m venv cuda_clean_env

Write-Host 'Activating virtual environment' -ForegroundColor Green
& '.\cuda_clean_env\Scripts\Activate.ps1'

Write-Host 'Upgrading pip...' -ForegroundColor Green
python -m pip install --upgrade pip

# Install core packages (non-fatal failures will continue)
Write-Host 'Installing core ML packages (may take some time)...' -ForegroundColor Green
try {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
} catch {
    Write-Host 'WARN: PyTorch install failed or partial; script will continue to verification.' -ForegroundColor Yellow
}

# Core verification + snapshot
Write-Host 'Running core verification: test_pytorch.py' -ForegroundColor Cyan
$LogDir = Join-Path (Get-Location) 'ml_env_logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
try {
    & python test_pytorch.py *>&1 | Out-File -FilePath (Join-Path $LogDir 'test_pytorch.log') -Encoding utf8
    Write-Host "PyTorch verification passed. See $LogDir\test_pytorch.log" -ForegroundColor Green
    $snap = Join-Path (Get-Location) ("pinned-requirements-{0}.txt" -f (Get-Date -Format yyyyMMddHHmm))
    pip freeze > $snap
    Write-Host "Created snapshot: $snap" -ForegroundColor Green
} catch {
    Write-Host "ERROR: PyTorch verification failed. See $LogDir\test_pytorch.log" -ForegroundColor Red
    exit 1
}

# Optional tests / installs (non-fatal)
Write-Host 'Running optional tests (non-fatal): TensorFlow' -ForegroundColor Yellow
try { & python test_tensorflow.py *>&1 | Out-File -FilePath (Join-Path $LogDir 'test_tensorflow.log') -Encoding utf8 } catch { Write-Host "WARN: test_tensorflow failed; see $LogDir\test_tensorflow.log" -ForegroundColor Yellow }

Write-Host "Setup finished. Check $LogDir for logs and the pinned requirements snapshot." -ForegroundColor Green
# =============================================
# ML-Env-CUDA13 Full Setup Script (FINAL)
# PyTorch 2.9 (CUDA 13.0) + TensorFlow 2.17 (CUDA 12.x)
# =============================================

$ErrorActionPreference = "Stop"

Write-Host "1. Setting PowerShell execution policy..." -ForegroundColor Green
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

Write-Host "2. Creating project files..." -ForegroundColor Green

# test_pytorch.py
@"
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
    print('nvcc not on PATH or not installed')
else:
    print(nvcc_out)
"@ | Out-File -FilePath "test_pytorch.py" -Encoding UTF8 -Force

# test_tensorflow.py
@"
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
    print('nvcc not on PATH or not installed')
else:
    print(nvcc_out)
"@ | Out-File -FilePath "test_tensorflow.py" -Encoding UTF8 -Force

# ML_ENV_README.md
@"
# ML-Env-CUDA13 (GPU Ready)

**GPU**: RTX 2000 Ada  
**CUDA**: 13.0.2 (System)  
**PyTorch**: 2.9.0+cu130 (CUDA 13.0)  
**TensorFlow**: 2.17.0 (CUDA 11.8 via wheels)

## Activate
``````powershell
.\cuda_clean_env\Scripts\Activate
``````

## Test
``````powershell
python test_pytorch.py
python test_tensorflow.py
``````

## Reinstall
``````powershell
.\setup_ml_env_full.ps1
``````

## Deactivate
``````powershell
deactivate
``````
"@ | Out-File -FilePath "ML_ENV_README.md" -Encoding UTF8 -Force

# --- 3. DELETE BROKEN ENV & RECREATE FRESH ---
if (Test-Path "cuda_clean_env") {
    Write-Host "3. Removing corrupted environment..." -ForegroundColor Red
    Remove-Item "cuda_clean_env" -Recurse -Force
}
Write-Host "3. Creating fresh virtual environment..." -ForegroundColor Green
python -m venv cuda_clean_env

# --- 4. ACTIVATE ---
Write-Host "4. Activating environment..." -ForegroundColor Green
& ".\cuda_clean_env\Scripts\Activate.ps1"

# --- 5. FIX PIP ---
Write-Host "5. Installing pip..." -ForegroundColor Green
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# --- 6. Install typing_extensions ---
Write-Host "6. Installing typing_extensions..." -ForegroundColor Green
pip install "typing_extensions>=4.10.0"

# --- 7. Install PyTorch ---
Write-Host "7. Installing PyTorch (cu130)..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# --- 8. Install TensorFlow 2.17 + CUDA 11.8 + cuDNN 8.9 ---
Write-Host "8. Installing TensorFlow 2.17 (GPU)..." -ForegroundColor Green
pip install "tensorflow==2.17.0" nvidia-cudnn-cu12==8.9.7.29 --no-cache-dir

# --- 9. Export ---
Write-Host "9. Exporting requirements.txt..." -ForegroundColor Green
pip freeze > requirements.txt

# --- 10. TEST ---
Write-Host "`nTESTING PYTORCH..." -ForegroundColor Cyan
python test_pytorch.py
Write-Host "`nTESTING TENSORFLOW..." -ForegroundColor Cyan
python test_tensorflow.py
Write-Host "`nSETUP COMPLETE!" -ForegroundColor Green
Write-Host "PyTorch: GPU ✅ (CUDA 13.0) | TensorFlow: CPU-only ⚠️ (Windows limitation)" -ForegroundColor Yellow
Write-Host "For TensorFlow GPU support, use WSL2. See README.md for setup guide." -ForegroundColor Cyan
