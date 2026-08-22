# Post-Mortem: CI Pipeline `[Errno 28] No space left on device` (Commit `612c0c8`)

## Incident Summary
During the execution of the GitHub Actions CI pipeline, the build jobs consistently failed with an `[Errno 28] No space left on device` (OOM / Disk Space Exhaustion) error precisely during the `pip install` step while downloading/unpacking the `torch` wheel (`/tmp/.../torch/lib/libshm`).

## Root Cause
The root cause of the disk space exhaustion was `pip` defaulting to the ~2.5GB CUDA-enabled (GPU) version of PyTorch instead of the much smaller ~200MB CPU-only version. 

Although `torch --index-url https://download.pytorch.org/whl/cpu` was specified in the `requirements.txt` / `requirements_docker.txt`, `pip` often ignores inline index URLs in certain environments or when parsing requirements sequentially, opting instead for the default PyPI registry index which serves the bulky CUDA wheels.

Downloading and extracting the multi-gigabyte library overwhelmed the limited ephemeral runner disk space allocated by GitHub Actions for that job layer, causing an immediate crash.

## Resolution
To ensure that `pip` strictly adheres to the CPU-only PyTorch distribution and does not fallback to PyPI's CUDA releases, the dependency manifests were refactored:

1. **Global Index Declaration**: `--extra-index-url https://download.pytorch.org/whl/cpu` was moved to the very first line of both `requirements.txt` and `requirements_docker.txt`. This forces `pip` to check the PyTorch registry *before* PyPI for all subsequent packages in the file.
2. **Strict Version Pinning**: The `torch` dependency was explicitly pinned to the CPU wheel: `torch==2.2.2+cpu`. 

This combination logically guarantees the environment resolves and downloads the lightweight `+cpu` wheel, bringing the PyTorch footprint down to ~200MB and allowing the CI pipeline to complete successfully without running out of disk space or memory.
