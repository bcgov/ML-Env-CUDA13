# ML-Env-CUDA13
**GPU-Accelerated Python ML Environment**

---

> Observed during verification (example run):
>
> - GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
> - nvcc (WSL): 12.6 (release 12.6, V12.6.85)
> - Host driver (Windows / nvidia-smi): 581.42 (reports CUDA Version: 13.0)
> - PyTorch: 2.9.0+cu128 (runtime tag; cudnn 91002)
> - TensorFlow: 2.20.0 (build reports cuda_build 12.5.1; cudnn build 9)
> - Optional libs observed: xformers 0.0.33.post1; llama-cpp-python import OK
>
> Note: minor-version differences between wheel tags, runtime CUDA, `nvcc`, and the host driver are common. Use the core test logs in `ml_env_logs/` as the canonical verification artifacts.


## 1. Project Overview

- Automated, reproducible, GPU-accelerated Python environment for deep learning and CUDA development
- Optimized for NVIDIA RTX 2000 Ada Generation Laptop GPU, but fully compatible with any CUDA 13.0+ GPU (RTX 30xx, 40xx, 50xx, Professional series, etc.)
- PyTorch and TensorFlow support (see environment matrix below)
- One-click setup scripts for Windows and WSL2/Ubuntu

---

## 2. GPU Support Matrix

## 2. GPU Support Matrix

| Environment         | PyTorch GPU | TensorFlow GPU | CUDA Version | Notes |
|--------------------|:-----------:|:--------------:|:------------:|-------|
| Windows Native     |     ✅      |      ❌        |   13.0       | Only PyTorch supports GPU (System CUDA). TensorFlow is CPU-only. **Legacy support only.** |
| WSL2 + Ubuntu      |     ✅      |      ✅        |   12.x **OR** 13.0 | **Primary Env.** Supports stable (`cu126`) or Nightly (`cu130` via `--cuda13` flag). TF works in both. |

---

## 3. Common Project Structure & Usage

```
ML-Env-CUDA13/
├── docs/                          ← Project Documentation & Reports
│   ├── adr/                       ← Architectural Decision Records
│   └── ml_env_cuda13_upgrade_report.md
├── scripts/
│   ├── setup_ml_env_full.ps1      ← Windows setup script (Legacy/Limited)
│   └── setup_ml_env_wsl.sh        ← WSL2/Ubuntu setup script (Primary)
├── tests/                         ← Verification scripts (pytorch, tensorflow, etc.)
├── cuda_clean_env/                ← Local virtual environment (Windows Only - Legacy)
├── ml_env_logs/                   ← Logs
├── archive/                       ← Old pinned requirements
├── requirements.in                ← Application application dependencies (WSL)
├── requirements-dev.in            ← Dev tools (WSL)
├── requirements.txt               ← Pinned application lockfile (WSL)
├── README.md                      ← Main documentation
```

**Note:** On WSL, the environment is created at `~/ml_env` (outside the repo) to avoid file permission issues.

- Run tests:
   - Core GPU verification (required):
      - WSL: `python tests/test_torch_cuda.py`
      - Windows/PowerShell: `python tests/test_pytorch.py`
   - Phase 4 Dependencies (GGUF/SentencePiece):
      - WSL: `python tests/test_phase4_deps.py`
   - Additional checks (optional): `python tests/test_tensorflow.py`, `python tests/test_xformers.py`, `python tests/test_llama_cpp.py`
   - Regenerate test scripts without installing packages:
      - WSL: `bash scripts/setup_ml_env_wsl.sh --regen-tests-only`
      - PowerShell: `./scripts/setup_ml_env_full.ps1 -RegenTestsOnly`
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
   .\scripts\setup_ml_env_full.ps1
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
- PyTorch: GPU acceleration (CUDA 12.6 or CUDA 13.0 Nightly)
- TensorFlow: GPU acceleration (TensorFlow 2.x)


#### Initial Setup Steps
1. Enable Windows Features (WSL, Virtual Machine Platform) & Install Ubuntu.
2. Run bash setup script:
   *   **Standard (Stable):**
       ```bash
       bash scripts/setup_ml_env_wsl.sh
       ```
   *   **CUDA 13 (Nightly):**
       ```bash
       bash scripts/setup_ml_env_wsl.sh --cuda13
       ```
3. Activate environment:
   ```bash
   source "$HOME/ml_env/bin/activate"
   ```

#### If you already ran the installer and the venv exists, activate it:
  ```bash
  source ~/ml_env/bin/activate
  ```

PyTorch and TensorFlow both generally use CUDA 12.x in WSL2, but the minor-version components reported by different tools can differ. This is expected; wheel build tags, TensorFlow's build-info, the system nvcc, and the Windows host driver do not always report the same minor version.

Observed examples from test runs:

- PyTorch runtime: 2.9.0+cu128 (runtime CUDA ~12.8; cudnn 91002)
- TensorFlow build: 2.20.0 (build reports CUDA 12.5.1)
- nvcc (toolkit on WSL): 12.6 (nvcc release 12.6, V12.6.85)
- Host driver (Windows): reports CUDA Version 13.0 via `nvidia-smi` (driver 581.42)
- Pinned requirements snapshot: `pinned-requirements-<YYYYMMDDHHMM>.txt` — created by the installer after the core torch gate passes; include this file when reporting or opening PRs for reproducibility.

Notes and guidance:

- Minor mismatches between build-time CUDA, runtime CUDA, and system nvcc are common and usually benign as long as the runtime reports GPU availability.
- **Performance Tip:** Always clone repositories and create virtual environments within the Linux native filesystem (e.g., `~/repos/`) rather than the Windows mount (`/mnt/c/...`). Accessing `/mnt/c/` from WSL2 introduces significant file I/O overhead which slows down git operations and python imports.
- The definitive verification is the core GPU test(s) included in this repo — the installer writes their output into `ml_env_logs/` and only snapshots the environment (`pinned-requirements-<timestamp>.txt`) after the core gate passes.
- Important artifacts to consult or attach to PRs:
   - `ml_env_logs/test_torch_cuda.log` (WSL) or `ml_env_logs/test_pytorch.log` (PowerShell): core gate logs
   - `ml_env_logs/test_tensorflow.log`: TensorFlow verification output
   - `pinned-requirements-<YYYYMMDDHHMM>.txt`: pip freeze created after a passing core gate

If the core gate fails, follow the Troubleshooting section and re-run the setup after addressing the problem (drivers, nvcc, or package wheel compatibility).

---

## 5. Dependency Management (New Hybrid Policy)

This repository follows a **Hybrid Dependency Policy** (ADR 001):

1.  **Foundation Layer (Exceptions):** `torch`, `tensorflow`, and `nvidia-*` packages are managed dynamically by the `setup_ml_env_wsl.sh` script to handle detailed hardware/driver matching.
2.  **Application Layer:** All other packages are managed via `pip-tools` and `requirements.in`.

### How to add a package (WSL)
1.  Edit `requirements.in` (add your package name).
2.  Run the setup script again:
    ```bash
    bash scripts/setup_ml_env_wsl.sh
    ```
    *   It will automatically compile `requirements.txt` and sync your environment.
    *   **Automatic Refresh:** The script now automatically recompiles `requirements.txt` on every run to ensure the Application Layer remains consistent with the Foundation Layer (e.g., preventing Torch version conflicts).
    *   Do **not** edit `requirements.txt` manually.

### Windows Native
*   Legacy Mode: Continue using `pip install` and `pip freeze > requirements.txt` for now.

---

## 6. llama.cpp Integration (Sibling Repository)

This environment is designed to work alongside `llama.cpp` for GGUF model conversion and inference.

**Repository:** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

### Required Setup
```bash
# Clone as sibling directory
cd ..  # Go to parent of ML-Env-CUDA13
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
./build/bin/llama-cli --version  # Verify build
```

### Build Artifacts
| Binary | Location | Purpose |
|--------|----------|---------|
| `llama-cli` | `build/bin/llama-cli` | Standalone CLI for model inference |
| `llama-server` | `build/bin/llama-server` | OpenAI-compatible HTTP API server |
| `llama-quantize` | `build/bin/llama-quantize` | GGUF model quantization |
| `convert_hf_to_gguf.py` | Root directory | HuggingFace → GGUF conversion |

### Shared Libraries (CUDA Support)
| Library | Purpose |
|---------|---------|
| `libggml-cuda.so` | CUDA GPU acceleration |
| `libllama.so` | Core inference library |
| `libggml.so` | Base tensor operations |

### Important Notes
*   **Linux ELF 64-bit:** Output binaries are Linux ELF, NOT Windows .exe.
*   **WSL2:** Must execute within WSL2 environment.
*   **Drivers:** GPU acceleration requires NVIDIA drivers accessible in Linux environment.
*   **LD_LIBRARY_PATH:** Ensure `LD_LIBRARY_PATH` includes `build/bin/` for shared library resolution.
*   **Build time:** 5-15 minutes (one-time unless llama.cpp is updated).

### Verification
```bash
ls ../llama.cpp/build/bin/llama-cli
../llama.cpp/build/bin/llama-cli --version
```

---

## 7. Troubleshooting (Common)

| Issue | Fix |
|------|-----|
| `Activate.ps1` blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `CUDA not available` | Check `nvidia-smi`, restart PC |
| `ImportError: undefined symbol: ncclDevCommDestroy` | (Fixed) Run `setup_ml_env_wsl.sh` again. The script now purges conflicting `nvidia-*-cu12` packages. |
| Install fails | Run `python -m pip install ...` |

---

## 8. Advanced Topics / Next Steps
- VS Code workspace integration
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
2. NVIDIA Driver 581.42 → Reboot
3. CUDA 13.0.2 → Custom Install (nvcc, Runtime, Samples)

2. Install Ubuntu from Microsoft Store or PowerShell
3. Launch Ubuntu and update system

> Verify after install:
> ```powershell
> # On the Windows host (PowerShell) - driver and WSL status
> nvidia-smi
> wsl --list --verbose
> ```
> 
> ```bash
> # In WSL/Ubuntu (run inside the distro) - toolkit and runtime checks
> nvcc --version || echo "nvcc not installed in WSL"
> nvidia-smi || echo "nvidia-smi not available in this WSL instance"
> 
> # Quick Python verification (WSL or Windows venv)
> python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), getattr(torch.version, "cuda", None))'
> python -c 'import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices("GPU"))'
> ```

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
