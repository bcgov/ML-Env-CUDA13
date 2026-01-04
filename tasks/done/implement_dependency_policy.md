# Implement "Hybrid" Sealed Dependency Policy

**Status:** Backlog
**Priority:** High (Governance)
**Related Analysis:** [Dependency Policy Feasibility](../analysis/dependency_policy_feasibility.md)

## Context
We need to increase the rigour of dependency management in `ML-Env-CUDA13`. Following the analysis of the "Sanctuary" policy (ADR 073), we will implement a **Hybrid Approach**:
1.  **Foundation Layer:** `setup_ml_env_` scripts continue to manage dynamic hardware-specific foundation packages (PyTorch, TensorFlow, CUDA).
2.  **Application Layer:** All upper-level dependencies (e.g., `transformers`, `fastapi`, dev tools) must be managed via `pip-tools` (`.in` -> `.txt`) to ensure reproducibility.

## Work Plan

### 1. Governance (The ADR)
- [ ] Draft a new Architecture Decision Record (ADR) for *this* project.
    -   **Title:** `docs/adr/001-hybrid-dependency-management.md`
    -   **Content:** Adapt the principles of "Sanctuary ADR 073" but explicitly formalize the "Foundation Exception" for CUDA/Hardware isolation.
    -   **Goal:** Define the rules for "when to pin" vs "when to let the script decide".

### 2. Tooling Setup
- [ ] Add `pip-tools` to the initial setup scripts (after Python installation).
- [ ] Create `Makefile` (or `Taskfile`) aliases for common operations:
    -   `make lock`: Run `pip-compile`.
    -   `make sync`: Run `pip-sync` (or install) for the application layer.

### 3. Migration
- [ ] **Phase 1: Foundation Separation**
    -   Identify which packages in `requirements-wsl.txt` are "Foundation" (Torch, TF, Nvidia-*) vs "Application" (rich, markdown, etc.).
- [ ] **Phase 2: Create Intent Files**
    -   Create `requirements.in` for the Application layer.
    -   Create `requirements-dev.in` for testing/dev tools.
- [ ] **Phase 3: Update Scripts**
    -   Modify `setup_ml_env_wsl.sh`:
        1.  Install System/Foundation (as today).
        2.  `pip install pip-tools`.
        3.  `pip-compile requirements.in` (if `.txt` missing or stale request).
        4.  `pip install -r requirements.txt` (Application Layer).

### 4. Verification
- [ ] Verify that `torch` version remains controlled by the CLI flags (`cu126`/`cu130`) and is NOT overridden by `pip-tools`.
- [ ] Verify `requirements.txt` generation works on both WSL and Windows.

## Acceptance Criteria
- [ ] ADR 001 created and approved.
- [ ] `setup_ml_env_wsl.sh` uses `pip-tools` for non-GPU dependencies.
- [ ] A clear separation exists between "Hardware Foundation" and "Software Application".
