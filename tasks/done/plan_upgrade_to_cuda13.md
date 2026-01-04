# Plan: Upgrade WSL2 Environment to CUDA 13.0

**Status:** Backlog  
**Parent Task:** [Explore CUDA 13 Runtime for WSL2 Environment](./explore_cuda_13_wsl.md)  

## 1. Analysis
The current repository is in a split state regarding CUDA 13:
*   **Windows (`setup_ml_env_full.ps1`):** Already attempts to use `https://download.pytorch.org/whl/cu130`.
*   **WSL2 (`setup_ml_env_wsl.sh`):** Defaults to `cu126` and uses the standard stable release index (`https://download.pytorch.org/whl/${CUDA_TAG}`).

To support CUDA 13.0 on WSL2, we must switch to **PyTorch Nightly** builds, as `cu130` is not yet in the stable channel for Linux/WSL.

## 2. Implementation Plan

### 2.1 Script Modifications (`setup_ml_env_wsl.sh`)
Refactor the setup script to support a "nightly" or "cuda13" mode.

1.  **Add CLI Flag:**
    *   Add `--cuda13` or `--nightly` argument to the script.
2.  **Conditional Index & Flags:**
    *   If `--cuda13` is set:
        *   Set `CUDA_TAG="cu130"`
        *   Set `PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"`
        *   **Important:** Add `--pre` flag to the `pip install` command (required for nightly builds).
    *   Else (Default):
        *   Keep existing `cu126` / auto-detect behavior.
        *   Keep standard index.
3.  **Post-Install Checks:**
    *   Ensure `test_torch_cuda.py` output parsing handles potential nightly version strings (e.g., `2.10.0.dev20250104+cu130`).

### 2.2 Verification Plan
1.  **Dry Run:**
    *   Run `bash setup_ml_env_wsl.sh --cuda13 --regen-tests-only` to verify templates.
2.  **Install Test:**
    *   Run `bash setup_ml_env_wsl.sh --cuda13`.
    *   Monitor for dependency conflicts (TensorFlow might conflict if installed in the same environment; consider making TensorFlow optional or CPU-only in this mode).
3.  **Runtime Verification:**
    *   Run `python test_torch_cuda.py`.
    *   Expected Output: `torch.version.cuda` should be `13.0`.

## 3. Risk Assessment
*   **TensorFlow Compatibility:** TensorFlow likely does not have a `cu130` wheel yet.
    *   *Mitigation:* In CUDA 13 mode, warn user that TensorFlow might be CPU-only or downgraded to CUDA 12 compat mode (if possible).
*   **Stability:** Nightly builds can be unstable.
    *   *Mitigation:* Keep this as an *optional* flag, not the default, until stable release.

## 4. Work Checklist
- [ ] Create and push `feature/wsl-cuda13` branch to GitHub.
- [ ] Modify `setup_ml_env_wsl.sh` to accept `--cuda13`.
- [ ] Update `pip install` logic to handle `--pre` and nightly index.
- [ ] Test installation on WSL2.
- [ ] Update `README.md` to document the new experimental mode.
