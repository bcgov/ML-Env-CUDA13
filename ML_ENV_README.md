# ML-Env-CUDA13 (Quick Reference)

**GPU**: NVIDIA RTX 2000 Ada Generation Laptop GPU
**Supported Modes**:
*   **Stable**: CUDA 12.6 (PyTorch 2.9 stable)
*   **Nightly**: CUDA 13.0 (PyTorch 2.9/2.10 Nightly) via `--cuda13`

## Activate
```bash
source "$HOME/ml_env/bin/activate"
```

## Test
```bash
# Core Verification
python tests/test_torch_cuda.py

# Application Verification
python tests/test_llama_cpp.py

# Phase 4 Verification (GGUF/SentencePiece)
python tests/test_phase4_deps.py
```

## Reinstall / Update
```bash
# Standard (Stable)
bash scripts/setup_ml_env_wsl.sh

# CUDA 13 (Nightly)
bash scripts/setup_ml_env_wsl.sh --cuda13
```

## Deactivate
```bash
deactivate
```
