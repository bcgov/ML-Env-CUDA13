# Plan: Refactor `setup_ml_env_wsl.sh`

## Goal
Update the WSL setup script to support:
1.  **New Directory Structure:** Tests are now in `tests/`.
2.  **Hybrid Dependency Policy:**
    *   Foundation (Torch/TF) installed via script logic.
    *   Application (everything else) installed via `pip-tools` from `requirements.in`.
3.  **CUDA 13 Support:** Add `--cuda13` flag to switch to PyTorch Nightly `cu130`.

## Changes

### 1. CLI Arguments
Update argument parsing to handle `--cuda13`.
*   If set: `CUDA_TAG="cu130"`, `PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"`.
*   Also set `PIP_PRE_FLAG="--pre"` for nightly.

### 2. Test Generation (`regen_write_tests`)
*   Update output paths to `tests/test_torch_cuda.py`, etc.
*   Ensure `tests/` directory exists.

### 3. Foundation Installation
*   Keep existing logic for `torch`, `torchvision`, `torchaudio`.
*   Apply `PIP_PRE_FLAG` if set.
*   Keep `tensorflow` logic (CPU-only or stable GPU depending on compat).

### 4. Application Layer (New)
*   Install `pip-tools`.
*   Compile `requirements.in` -> `requirements.txt` (if missing or forced).
*   Install `requirements.txt`.
*   Install `requirements-dev.in` -> `requirements-dev.txt` (optional/default).

### 5. Verification
*   Update verification steps to call `python tests/test_torch_cuda.py`.

## Script Structure
```bash
# ... CLI parsing ...

# ... regen_write_tests (to tests/) ...

# ... System Deps ...

# ... Venv Creation ...

# ... Foundation Layer (Torch/TF) ...
pip install $PIP_PRE_FLAG torch --index-url $PYTORCH_INDEX

# ... Application Layer ...
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
pip-compile requirements-dev.in
pip install -r requirements-dev.txt

# ... Verification (tests/...) ...
```
