# Feasibility Analysis: ADR 073 for ML-Env-CUDA13

**Status:** Draft Evaluation
**Date:** 2026-01-04
**Context:** Reviewing [ADR 073 Compliance](../INBOX/073_standardization_of_python_dependency_management_across_environments.md) for the GPU-optimized ML environment.

## 1. Executive Summary
**Recommendation:** **Conditional Feasibility (Hybrid Approach)**

Strict adherence to ADR 073 (using `pip-compile` to generate lockfiles) is **feasible and recommended** for controlling the `ML-Env-CUDA13` environment, but handling the **hardware-specific variance** (CUDA 12 vs 13) requires a modification to the standard single-file pattern. We cannot use a single `requirements.txt` for all variations; we must generate specific profiles (e.g., `requirements-wsl-cuda12.txt`).

## 2. Gap Analysis

| Feature | ADR 073 Policy | Current ML-Env Status | Gap / Conflict |
| :--- | :--- | :--- | :--- |
| **Source of Truth** | `.in` file (Intent) | Setup script logic + manual `pip install` | **High**: Script decides what to install at runtime. |
| **Lock Mechanism** | `pip-compile` (Static Resolution) | `pip freeze` (Dynamic Snapshot) | **High**: We currently "install then freeze" rather than "compile then install". |
| **Parity** | "One Runtime World" | conditional `setup_ml_env_wsl.sh` | **Medium**: Script adapts to host (nvcc 12 vs 13). |
| **Nightly Builds** | Implicitly discouraged (instability) | Required for CUDA 13 (PyTorch Nightly) | **Medium**: Locking a nightly build is valid only for that specific day. |

## 3. The "Fine-Tuning" Challenge
The user specifically asked about **complex dependency chains for fine-tuning** (e.g., `unsloth`, `bitsandbytes`, `flash-attn`).

*   **The Problem:** These libraries often require compilation against the *exact* installed PyTorch version during installation, or fetching specific pre-built wheels matching the CUDA version.
*   **pip-compile limitation:** `pip-compile` resolves versions but does not usually handle the complex *build-time* constraints or dynamic wheel selection logic that `setup_ml_env_wsl.sh` currently performs.
*   **The Risk:** A statically generated `requirements.txt` might point to a `flash-attn` version that *should* work, but if installed via simple `pip install -r`, it might trigger a long build-from-source that fails due to missing system headers, whereas the setup script handles this prep.

## 4. Recommendations

### 4.1. Core Environment (The "Engine")
We should **NOT** treat `ML-Env-CUDA13` as a standard "Service". It is an **Infrastructure Bootstrap**.

*   **Exception Request:** Grant an exception for the *Bootstrap Layer* (PyTorch/TensorFlow core installation). The specific logic to match Driver <-> CUDA <-> Torch is best handled by the intelligent setup script (`setup_ml_env_wsl.sh`).
*   **Compliance Layer:** For all *other* packages (e.g., `transformers`, `datasets`, `fastapi`, `pydantic`), we **should** enforce ADR 073.

### 4.2. Proposed Workflow (The "Layered Onion")
Instead of a monolithic `requirements.txt`, we define the environment in two phases:

1.  **Phase 1: The Hardware Substrate (Exception)**
    *   Managed by `setup_ml_env_wsl.sh`.
    *   Installs `torch`, `torchvision`, `tensorflow`.
    *   Justification: Requires dynamic hardware detection (CUDA 12 vs 13) and index selection.

2.  **Phase 2: The Application Layer (Compliant)**
    *   Managed via `pip-tools`.
    *   `requirements-core.in` -> `requirements-core.txt`.
    *   Contains `transformers`, `scikit-learn`, `api-server`.
    *   These packages install *on top* of the substrate.

### 4.3. Handling "Nightly" (CUDA 13)
For the [Backlog Task](../backlog/plan_upgrade_to_cuda13.md):
*   Do not attempt to lock nightly builds in a permanent `requirements.txt`.
*   Let the setup script handle the `--pre` installation of `torch`, then install the standard secured requirements on top.

### 4.4. Clarification on Version Pinning
**Answer to Query:** Yes, `.in` files *can* and *should* contain explicit version constraints when required by the platform.
*   **General Rule:** Avoid strict pins (e.g., `pandas==2.0.3`) to allow the resolver to find optimal solutions.
*   **ML Exception:** For this environment, high-level constraints are often mandatory.
    *   *Example:* `flash-attn>=2.5.0` (to ensure Ada architecture support).
    *   *Example:* `numpy<2.0.0` (if a specific library hasn't updated yet).
*   **Mechanism:** `pip-compile` respects these constraints as absolute laws when generating the lockfile.

## 5. Conclusion
**Is it feasible?**
Yes, but strict "single file" locking will break the multi-cuda capability.

**Do we need an exception?**
Yes: An exception is required for the **PyTorch/TensorFlow Foundation Layer**. The complex matrix of [OS + Driver + CUDA + Python] is too dynamic for a static `pip-compile` without maintaining 4-5 separate lockfiles.

**Next Steps:**
1.  Continue with `setup_ml_env_` scripts for the Foundation Layer.
2.  Adopt `pip-tools` for specific tasks/projects that *run* in this environment, treating the `ML-Env` provided torch/cuda as "System Packages" (using `pip-compile --unsafe-package torch` so it doesn't try to manage them).
