# ML-Env-CUDA13 Upgrade Report for Project Sanctuary

**Date:** 2026-01-04
**Status:** Ready for Verification
**Feature Branch:** `feature/wsl-cuda13`

## 1. Executive Summary
The `ML-Env-CUDA13` environment has been upgraded to support **CUDA 13 (PyTorch Nightly)** and implements a **Hybrid Dependency Policy** (ADR 001) that isolates the hardware foundation from the application layer. This ensures stability for the "Foundation" (Drivers/Torch) while providing strict reproducibility for the "Application" (Agents/LLMs).

## 2. Environment Specifications (Target)

| Component | Stable (Default) | Nightly (CUDA 13) | Notes |
|-----------|------------------|-------------------|-------|
| **CUDA Tag** | `cu126` | `cu130` | Auto-detected or forced via `--cuda13` |
| **Python** | `3.11` | `3.11` | Standardized on 3.11 |
| **PyTorch** | `2.x.x+cu126` | `2.x.x.dev+cu130` | Nightly required for CUDA 13 |
| **TensorFlow**| `2.x.x` | `2.x.x` (CPU/Compat) | TF support for cu130 is experimental |
| **xformers** | `0.0.x+cu126` | *Likely Incompatible* | Check `tests/test_xformers.py` |
| **llama-cpp** | `0.3.x` (cu12) | *Source Build Req* | May need `CUDACXX` env var for 13.0 |

*Note: Exact version numbers depend on the specific day's install (for Nightly) or the `requirements-wsl.txt` snapshot.*

## 3. Command Reference

### Installation
| Goal | Command | Description |
|------|---------|-------------|
| **Standard Install** | `bash scripts/setup_ml_env_wsl.sh` | Installs Stable (CUDA 12.6), generates locked `requirements.txt`. |
| **CUDA 13 Install** | `bash scripts/setup_ml_env_wsl.sh --cuda13` | Installs Nightly (CUDA 13.0), bypasses strict foundation locks. |
| **Regen Tests** | `bash scripts/setup_ml_env_wsl.sh --regen-tests-only` | Updates `tests/` scripts without installing packages. |

### Dependency Management (The Loop)
*   **Edit:** Modify `requirements.in` (High-level intents).
*   **Apply:** Run the setup script again (`bash scripts/setup_ml_env_wsl.sh`).
    *   It detects `requirements.in` change -> Compiles to `.txt` -> Syncs environment.
    *   *Safe to run repeatedly (Idempotent).*

## 4. Verification Commands

Run the following from the repo root (after activating `~/ml_env/bin/activate`):

```bash
# 1. Foundation Verification (Critical Gate)
python tests/test_torch_cuda.py

# 2. Application Verification (Optional)
python tests/test_llama_cpp.py
python tests/test_xformers.py
```

## 5. Known Issues & Warnings

### ⚠️ CUDA 13 / Nightly Stability
*   **PyTorch Nightly:** API changes may occur daily. `cu130` wheels are explicitly "Pre-release".
*   **xformers:** Pre-built wheels for `cu130` likely do not exist yet. `pip` may try to build from source or fail. If it fails, remove it from `requirements.in` for the Nightly environment.
*   **TensorFlow:** `tensorflow` stable wheels link against CUDA 12. They *might* run under CUDA 13 via forward compatibility, but CPU-only fallback is expected if drivers are mismatched.

### ⚠️ Breaking Changes
*   **Directory Move:**
    *   Setup scripts: `root/*.sh` -> `scripts/*.sh`
    *   Tests: `root/test_*.py` -> `tests/test_*.py`
    *   Pinned reqs: Moved to `archive/`
*   **Requirements Path:**
    *   Use `requirements.in` for adding new libraries.
    *   Do **NOT** manually edit `requirements.txt` (it is now machine-generated).

## 6. Next Steps for Project Sanctuary
1.  Clone `feature/wsl-cuda13`.
2.  Run `bash scripts/setup_ml_env_wsl.sh --cuda13`.
3.  Run `python tests/test_torch_cuda.py` to confirm CUDA 13 activation.
4.  Proceed with `forge-llm.md` documentation using these new paths.
