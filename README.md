# ML-Env-CUDA13
**GPU-Accelerated Python ML Environment**

---

## 1. Project Overview

- Automated, reproducible, GPU-accelerated Python environment for deep learning and CUDA development
- Optimized for NVIDIA RTX 2000 Ada Generation Laptop GPU, but fully compatible with any CUDA 13.0+ GPU (RTX 30xx, 40xx, 50xx, Professional series, etc.)
- PyTorch and TensorFlow support (see environment matrix below)
- One-click setup scripts for Windows and WSL2/Ubuntu

---

## 2. GPU Support Matrix

| Environment         | PyTorch GPU | TensorFlow GPU | CUDA Version | Notes |
|--------------------|:-----------:|:--------------:|:------------:|-------|
| Windows Native     |     ✅      |      ❌        |   13.0       | Only PyTorch supports GPU. TensorFlow is CPU-only after v2.10. |
| WSL2 + Ubuntu      |     ✅      |      ✅        |   12.x (PyTorch runtime 12.1; TF build 12.5.1; nvcc 12.6; host driver reports 13.0) | Both PyTorch and TensorFlow support GPU in WSL2. |

---

## 3. Common Project Structure & Usage

```
ML-Env-CUDA13/
├── setup_ml_env_full.ps1      ← Windows setup script
├── setup_ml_env_wsl.sh        ← WSL2/Ubuntu setup script
├── cuda_clean_env/            ← Virtual environment (Windows)
├── test_pytorch.py            ← GPU test
├── test_tensorflow.py         ← GPU test
├── requirements.txt           ← Exact package versions
├── ML_ENV_README.md           ← Quick reference
└── README.md                  ← Main documentation
```

- Run tests: `python test_pytorch.py` and `python test_tensorflow.py`
- Train models: `python train.py`, `python inference.py`
- Troubleshooting: See section below

---
## 4. Environment options
### 4.1 Windows Native Environment

#### Features
- PyTorch: GPU acceleration (CUDA 13.0)
- TensorFlow: CPU-only (no GPU after v2.10)

#### Initial Setup Steps
1. Install dependencies (Visual Studio, NVIDIA Driver, CUDA Toolkit)
2. Run PowerShell setup script:
***NOTE:**  you likely might need to turn off cisco to have libraries install
   ```powershell
   .\setup_ml_env_full.ps1
   ```
3. Activate environment:
   ```powershell
   .\cuda_clean_env\Scripts\Activate
   ```
4. Run tests and train models as above

#### If you already ran the installer and the venv exists, activate it:
   ```powershell
   .\cuda_clean_env\Scripts\Activate
   ```

#### Limitations
- TensorFlow GPU not supported after v2.10 *(temporary; future Windows releases may restore support, but currently use WSL2 for TensorFlow GPU)*
- PyTorch uses system CUDA 13.0

---

### 4.2. WSL2 + Ubuntu Environment

#### Features
- PyTorch: GPU acceleration (CUDA 12.x — PyTorch runtime 12.1 observed)
- TensorFlow: GPU acceleration (TensorFlow build reports CUDA 12.5.1 in current tests)

#### Initial Setup Steps
1. Enable Windows Features:
   - `Virtual Machine Platform` and `Windows Subsystem for Linux`
   - Reboot
2. Install Ubuntu for WSL2:
   ```powershell
   wsl --install -d Ubuntu
   wsl --set-default-version 2
   ```
3. Launch Ubuntu and create user account
4. Run bash setup script:
***NOTE:**  you likely might need to turn off cisco to have libraries install
   ```bash
   bash setup_ml_env_wsl.sh
   ```
5. Activate environment:
   ```bash
   source ~/ml_env/bin/activate
   ```
6. Run tests and train models as above

#### If you already ran the installer and the venv exists, activate it:
  ```bash
  source ~/ml_env/bin/activate
  ```

- PyTorch and TensorFlow both use CUDA 12.x in WSL2, but minor version components can differ between the frameworks, the system compiler, and the host driver. In current tests we observed:

- PyTorch runtime: 12.1 (built as `+cu121`)
- TensorFlow build: 12.5.1
- nvcc (toolkit on WSL): 12.6
- Host driver (Windows): reports CUDA Version 13.0 via `nvidia-smi` (driver 581.42)

Minor mismatches between build-time CUDA, runtime CUDA, and system nvcc are common; always consult the test outputs (the setup script prints a JSON-like verification block) or `requirements-wsl.txt` for the exact versions present in your environment.
 - Full GPU support for both libraries

---

## 5. Troubleshooting (Common)

| Issue | Fix |
|------|-----|
| `Activate.ps1` blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `CUDA not available` | Check `nvidia-smi`, restart PC |
| Install fails | Run `python -m pip install ...` |
| Python 3.11 not found (WSL2) | Add deadsnakes PPA and update package list |

---

## 6. Advanced Topics / Next Steps
- VS Code workspace integration
- Model training templates
- CUDA C++ + Python bridge

---

## Software Dependencies (Install First)

> **Warning: These must be installed BEFORE running any script.**

| # | Dependency | Download / Install Link | Notes |
|---|------------|--------------------------|-------|
| 1 | **Visual Studio 2022 Community** | [https://visualstudio.microsoft.com/vs/community/](https://visualstudio.microsoft.com/vs/community/) | Select **"Desktop development with C++"** workload (Windows only) |
| 2 | **NVIDIA Driver 581.42** | [https://www.nvidia.com/Download/driverResults.aspx/219876/en-us/](https://www.nvidia.com/Download/driverResults.aspx/219876/en-us/) | File: `581.42-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe` (Windows host, required for WSL2 GPU) |
| 3 | **CUDA Toolkit 13.0.2** | [https://developer.nvidia.com/cuda-13.0.2-download-archive](https://developer.nvidia.com/cuda-13.0.2-download-archive) | Select: **Windows → x86_64 → 10/11 → exe (network)** (Windows only) |
| 4 | **Windows Subsystem for Linux (WSL2)** | [WSL2 Install Guide](https://learn.microsoft.com/en-us/windows/wsl/install) | Enable via Windows Features or PowerShell. Required for Ubuntu/TensorFlow GPU. |
| 5 | **Ubuntu (WSL2)** | [Microsoft Store: Ubuntu](https://aka.ms/wslubuntu) | Recommended: Latest LTS (e.g., 22.04 or 24.04). |

### Installation Order

**Windows Native:**
1. Visual Studio 2022 → Reboot if prompted
2. NVIDIA Driver 581.42 → Reboot
3. CUDA 13.0.2 → Custom Install (nvcc, Runtime, Samples)

**WSL2/Ubuntu:**
1. Enable WSL2 (Windows Features or PowerShell)
2. Install Ubuntu from Microsoft Store or PowerShell
3. Launch Ubuntu and update system

> Verify after install:
> ```powershell
> nvidia-smi
> nvcc --version
> wsl --list --verbose
> ```

---

## Automated Requirements File Generation

Both setup scripts now automatically generate the correct requirements file for your environment:

- **Windows Native:**
  - `setup_ml_env_full.ps1` generates `requirements.txt`
- **WSL2/Ubuntu:**
  - `setup_ml_env_wsl.sh` generates `requirements-wsl.txt`

**Always use the requirements file generated by your setup script for reproducibility.**

---

## Environment-Specific Requirements

To ensure reproducibility and compatibility, use separate requirements files for each environment:

- **Windows Native:**
  - Use `requirements.txt`
  - Generated from your Windows Python environment
  - Example:
    ```powershell
    pip freeze > requirements.txt
    ```

- **WSL2/Ubuntu:**
  - Use `requirements-wsl.txt`
  - Generated from your WSL2 ML Python environment (after activating your venv)
  - Example:
    ```bash
    source ~/ml_env/bin/activate
    pip freeze > requirements-wsl.txt
    ```

**Instructions:**
- When recreating the environment, use the correct requirements file for your OS.
- Document this in your README so users know which file to use.

---

## License

This repository is licensed under the Apache License, Version 2.0.
See the full license text in the `LICENSE` file included at the repository root.

By contributing to this repository you agree to license your contributions
under the same license. See `CONTRIBUTING.md` for contribution guidance and
`CODE_OF_CONDUCT.md` for expected community behaviour.

---

Apache License (short notice)

© 2025 British Columbia — Licensed under the Apache License, Version 2.0.
See `LICENSE` for details and the full text. When reusing code from this
repository, include the Apache-2.0 header in source files as shown in
`test_pytorch.py` and `test_tensorflow.py`.
