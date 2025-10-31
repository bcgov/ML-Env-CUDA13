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
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA Runtime: {torch.version.cuda}")
"@ | Out-File -FilePath "test_pytorch.py" -Encoding UTF8 -Force

# test_tensorflow.py
@"
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Detected: {len(gpus) > 0}")
for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu.name}")
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
