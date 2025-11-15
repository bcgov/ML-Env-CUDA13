# ML-Env-CUDA13 (GPU Ready)

> NOTE: The commands and examples in this README assume you are running inside the WSL (Ubuntu) shell unless a command is explicitly marked as `PowerShell` (Windows host). Use the PowerShell examples when running on Windows host.

**GPU**: NVIDIA RTX 2000 Ada Generation Laptop GPU
**CUDA (nvcc)**: 12.6 (nvcc release 12.6, V12.6.85)
**Host driver (nvidia-smi)**: Driver 581.42 (reports CUDA Version: 13.0)
**PyTorch**: 2.9.0+cu128 (runtime tag shows cuda_build ~12.8; cudnn version 91002)
**TensorFlow**: 2.20.0 (build reports cuda_build 12.5.1; cudnn build 9)
**Optional libs**: xformers 0.0.33.post1; llama-cpp-python import OK

Note: Minor-version differences between wheel build tags, runtime CUDA, the system `nvcc`, and the host driver are common. The installer verifies GPU availability (core tests) and records logs in `ml_env_logs/` — use those logs as the definitive instrumentation for reproducibility.

## Activate
```bash
source "$HOME/ml_env/bin/activate"
```

## Test
```bash
# WSL core GPU verification (required)
python test_torch_cuda.py

# Core GPU verification (WSL bash or Windows host)
python test_pytorch.py

# TensorFlow GPU verification (WSL)
python test_tensorflow.py

# Optional (non-fatal) checks
python test_xformers.py
python test_llama_cpp.py
```

## Reinstall
```bash
bash setup_ml_env_wsl.sh
```

## Deactivate
```bash
deactivate
```
