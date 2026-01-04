# 001. Hybrid Dependency Management Strategy

**Status:** Accepted
**Date:** 2026-01-04
**Context:** Project Sanctuary Standardization (ADR 073)

## Context
The `ML-Env-CUDA13` repository serves as a foundational infrastructure component (the "Engine Room") for high-performance ML workflows. It faces a unique challenge:
1.  **Hardware Variance:** It must support multiple host configurations (Windows Driver 13.0 vs WSL Runtime 12.x) and potentially different CUDA build targets (cu126 vs cu130).
2.  **Reproducibility:** Upstream consumers (Agentic IDEs, RAG pipelines) require strict, deterministic environments to prevent "works on my machine" issues.

Traditional "single strict lockfile" approaches (ADR 073) fail here because the *base* packages (`torch`, `tensorflow`, `nvidia-*`) must dynamically match the runtime hardware/driver context, which cannot be easily captured in a single static `requirements.txt` without combinatorial explosion.

## Decision
We will adopt a **Hybrid Dependency Management Strategy** that splits the environment into two distinct layers with different governance rules.

### Layer 1: The Foundation (Exception to ADR 073)
*   **Scope:** `torch`, `torchvision`, `torchaudio`, `tensorflow`, and `nvidia-*` libraries.
*   **Mechanism:** Managed dynamically by `setup_ml_env_wsl.sh` (or `.ps1`).
*   **Policy:**
    *   The setup script is the Source of Truth.
    *   It detects the environment (or accepts flags like `--cuda13`).
    *   It selects the appropriate index (Stable vs Nightly) and build tags.
    *   **NO Lockfile:** We do *not* pin these to a static hash in the application layer, because they are effectively "System Packages" for this specialized environment.

### Layer 2: The Application (Compliant with ADR 073)
*   **Scope:** All other Python libraries (e.g., `transformers`, `fastapi`, `pydantic`, `bitsandbytes`, `jupyter`).
*   **Mechanism:** Managed via `pip-tools` (`.in` -> `.txt`).
*   **Policy:**
    *   **Intent (`requirements.in`):** Human-readable list of direct dependencies. Version constraints (e.g., `flash-attn>=2.0`) are allowed and encouraged.
    *   **Lock (`requirements.txt`):** Machine-generated, pinned, and hashed.
    *   **Installation:** Installed *after* the Foundation layer.
    *   **Isolation:** The lockfile generation *must* assume the Foundation is present (or use `--unsafe-package` to ignore foundation conflicts if necessary, though prefer layering).

## Detailed Workflow

1.  **Operation:**
    Run the setup script (which is now Idempotent and Automating):
    ```bash
    bash scripts/setup_ml_env_wsl.sh --cuda13
    ```

2.  **Internal Execution Logic:**
    The script performs the following steps automatically:
    *   **Phase A: Foundation (Exception Layer):** Detects flags (e.g., `--cuda13`) and installs the correct `torch`/`tensorflow` / `nvidia-*` base.
    *   **Phase B: Tooling:** Installs `pip-tools`.
    *   **Phase C: Application (Strict Layer):**
        *   Checks `requirements.in`.
        *   *Automatically* runs `pip-compile requirements.in` if `requirements.txt` is missing.
        *   Installs from the pinned `requirements.txt`.


## Consequences
*   **Positive:** We achieve reproducibility for the complex application stack while retaining the flexibility needed for the fragile GPU/Driver stack.
*   **Negative:** There is a theoretical risk that an Application package updates and becomes incompatible with the specific Foundation version installed.
*   **Mitigation:** `requirements.in` can include broad constraints relative to the foundation (e.g., `numpy<2` if Torch requires it).

## Compliance
This ADR formally grants an **Exception** to the "Single Lockfile" rule of ADR 073 for the items in the Foundation Layer only. All other dependencies must follow the standard locking procedure.
