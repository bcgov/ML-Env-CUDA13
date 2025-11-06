# Contributing to ML-Env-CUDA13

Thanks for your interest in contributing. This doc is a short, practical
guide for contributors working within BC Government repositories.

1. Code of conduct
   - Please read `CODE_OF_CONDUCT.md` before contributing.

2. Issues and feature requests
   - Open an issue describing the bug or feature with steps to reproduce
     and any relevant environment details (OS, Python version, CUDA/runtime).

3. Pull requests
   - Fork the repository (if you don't have push access), create a feature
     branch, make small, focused changes, and open a PR against `master`.
   - Include a clear title and description explaining the change.
   - Ensure your changes include tests where appropriate and update
     documentation (README or ML_ENV_README.md) when behavior changes.

4. Formatting and tests
   - Use the project's existing style. Run linters and tests locally.
   - This repository ships small verification scripts that exercise GPU
     functionality. Prefer running those first to validate environment
     setup:
     ```bash
     # Activate the environment (WSL)
     source ~/ml_env/bin/activate
     python test_pytorch.py
     python test_tensorflow.py
     ```
     On Windows (PowerShell):
     ```powershell
     .\cuda_clean_env\Scripts\Activate
     python test_pytorch.py
     python test_tensorflow.py
     ```
   - For additional automated checks, run `python -m pytest` (if there are
     pytest tests) and `flake8` when configured in the project.

5. Security and sensitive data
   - Do not include secrets, credentials, or personally-identifiable data
     in your commits. If you accidentally commit a secret, rotate it
     immediately and open an issue referencing the leak so maintainers can
     assist.

6. Licensing and ownership
   - By contributing, you agree to license your contributions under the
     repository's license (see `LICENSE`). If you do not have authority to
     license code on behalf of an organization, do not submit it.

Thanks — the maintainers will review PRs and provide feedback.
