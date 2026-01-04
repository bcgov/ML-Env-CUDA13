param(
    [switch]$RegenTestsOnly
)

$ErrorActionPreference = 'Stop'
$Root = (Get-Item -Path $PSScriptRoot).Parent.FullName
$TestsDir = Join-Path $Root "tests"

Write-Host "ML-Env-CUDA13 PowerShell setup" -ForegroundColor Green
Write-Host "Root: $Root" -ForegroundColor Gray
Write-Host "Tests: $TestsDir" -ForegroundColor Gray

if (-not (Test-Path $TestsDir)) {
    New-Item -ItemType Directory -Path $TestsDir | Out-Null
}

function Write-TestFiles {
    Write-Host 'Writing test templates to tests/' -ForegroundColor Cyan
    
    # test_pytorch.py
    $PytorchTestPath = Join-Path $TestsDir "test_pytorch.py"
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
'@ | Out-File -FilePath $PytorchTestPath -Encoding UTF8 -Force

    # test_tensorflow.py
    $TensorflowTestPath = Join-Path $TestsDir "test_tensorflow.py"
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
'@ | Out-File -FilePath $TensorflowTestPath -Encoding UTF8 -Force
}

if ($RegenTestsOnly) {
    Write-TestFiles
    Write-Host 'Regenerated tests.' -ForegroundColor Green
    exit 0
}

# Ensure tests exist (create if missing)
if (-not (Test-Path (Join-Path $TestsDir 'test_pytorch.py')) -or -not (Test-Path (Join-Path $TestsDir 'test_tensorflow.py'))) {
    Write-Host 'Creating missing test files' -ForegroundColor Yellow
    Write-TestFiles
}

# Safe venv recreation: remove folder
$VenvDir = Join-Path $Root "cuda_clean_env"
if (Test-Path $VenvDir) {
    Write-Host "Removing existing virtual environment 'cuda_clean_env'" -ForegroundColor Yellow
    Remove-Item $VenvDir -Recurse -Force
}
Write-Host "Creating virtual environment 'cuda_clean_env'" -ForegroundColor Green
python -m venv $VenvDir

Write-Host 'Activating virtual environment' -ForegroundColor Green
& (Join-Path $VenvDir "Scripts\Activate.ps1")

Write-Host 'Upgrading pip...' -ForegroundColor Green
python -m pip install --upgrade pip

# Install core packages
Write-Host 'Installing Foundation Layer (PyTorch cu130)...' -ForegroundColor Green
try {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
} catch {
    Write-Host 'WARN: PyTorch install failed or partial; script will continue to verification.' -ForegroundColor Yellow
}

# Install TensorFlow 2.17 (CPU only on Windows usually, but pinned here for parity)
Write-Host "Installing TensorFlow 2.17..." -ForegroundColor Green
pip install "tensorflow==2.17.0" "typing_extensions>=4.10.0"

# Export requirements using the new logic? 
# For Windows, we still strictly export to requirements.txt for now as per legacy behavior, 
# or we could adopt pip-tools here too. Sticking to simple export to match 'setup_ml_env_full.ps1' scope for now.
Write-Host "Exporting requirements.txt..." -ForegroundColor Green
$ReqPath = Join-Path $Root "requirements.txt"
pip freeze > $ReqPath

# Core verification + snapshot
Write-Host 'Running core verification: tests/test_pytorch.py' -ForegroundColor Cyan
$LogDir = Join-Path $Root 'ml_env_logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$TestFile = Join-Path $TestsDir "test_pytorch.py"
try {
    & python $TestFile *>&1 | Out-File -FilePath (Join-Path $LogDir 'test_pytorch.log') -Encoding utf8
    Write-Host "PyTorch verification passed. See $LogDir\test_pytorch.log" -ForegroundColor Green
    
    $ArchiveDir = Join-Path $Root "archive"
    if (-not (Test-Path $ArchiveDir)) { New-Item -ItemType Directory -Path $ArchiveDir | Out-Null }
    $snap = Join-Path $ArchiveDir ("pinned-requirements-{0}.txt" -f (Get-Date -Format yyyyMMddHHmm))
    pip freeze > $snap
    Write-Host "Created snapshot: $snap" -ForegroundColor Green
} catch {
    Write-Host "ERROR: PyTorch verification failed. See $LogDir\test_pytorch.log" -ForegroundColor Red
    exit 1
}

# Optional tests
Write-Host 'Running optional tests (non-fatal): TensorFlow' -ForegroundColor Yellow
$TFTestFile = Join-Path $TestsDir "test_tensorflow.py"
try { & python $TFTestFile *>&1 | Out-File -FilePath (Join-Path $LogDir 'test_tensorflow.log') -Encoding utf8 } catch { Write-Host "WARN: test_tensorflow failed; see $LogDir\test_tensorflow.log" -ForegroundColor Yellow }

Write-Host "Setup finished. Check $LogDir for logs." -ForegroundColor Green
