# Universal DLL Coverage Collector

Collect LLVM code coverage for instrumented deep‑learning libraries (currently PyTorch) by executing large corpora of Python scripts in a controlled, parallel, and fault‑tolerant workflow. Results are merged into `.profdata` for analysis.

## What this does

- Uses an instrumented container image (PyTorch 2.2.0 provided) to run Python inputs under coverage.
- Buckets Python inputs by time intervals to stage work in batches.
- For each bucket, runs a driver that:
  - Iterates each top‑level subdirectory and executes all `.py` files via `exec()` in‑process (tolerating exceptions and `SystemExit`).
  - Produces one `.profraw` per subdirectory and merges them into a single `.profdata` per bucket.
- Copies merged `.profdata` (and a log if present) back to the host.

## Repository layout

- `run.py` – CLI entrypoint to orchestrate collection inside Docker.
- `cov.py` – Core orchestration: bucketing, container lifecycle, copy‑in/out, and running drivers.
- `scripts/acetest_driver.py` – Orchestrates per‑bucket runs; launches `torch_driver.py` once per subdirectory and merges coverage.
- `scripts/torch_driver.py` – Executes all Python files within a directory using `exec()`; never fails the overall run.
- `dockerfile/torch-2.2.0-instrumented.Dockerfile` – PyTorch 2.2.0 image with Clang/LLVM coverage enabled.
- `build.sh` – Convenience build script for the provided images.

## Requirements

- Docker installed and running.
- Disk space for instrumented builds and coverage artifacts.
- Python 3.10+ on the host to run orchestration scripts.

Note: `llvm-profdata` is installed inside the instrumented container image and used there to merge coverage. If you run the drivers on the host, ensure `llvm-profdata` is available on your PATH.

## Build the container images

The PyTorch 2.2.0 instrumented image is provided. Build it with:

```bash
bash build.sh
```

This creates the image `ncsu-swat/torch-2.2.0-instrumented`.

## Prepare inputs

You can point to a directory of Python files, or a directory already organized into interval buckets like `0-60/`, `60-120/`, etc. If your target directory is not bucketed, the tool will bucket by file modification times (mtime).

Example target layout (not pre‑bucketed):

```
target/
  project_a/...
  project_b/...
  single.py
```

Example bucketed layout (pre‑categorized):

```
target/
  0-60/
    project_a/...
  60-120/
    project_b/...
```

## Quick start (PyTorch)

Run the collector for PyTorch 2.2.0:

```bash
python3 run.py \
  --dll torch \
  --ver 2.2.0 \
  --target /absolute/path/to/target \
  --output _result \
  --baseline acetest \
  --itv 60 \
  --num_parallel 16
```

Arguments:
- `--dll`: Currently only `torch` is wired through the collector.
- `--ver`: Must match a built image tag, e.g., `2.2.0` → `ncsu-swat/torch-2.2.0-instrumented`.
- `--target`: Directory containing `.py` files or interval buckets. The tool will bucket if needed.
- `--output`: Host directory for results (default `_result`).
- `--baseline`: Driver set to use; pass `acetest` to use `scripts/acetest_driver.py`.
- `--itv`: Interval in seconds for bucketing if `--target` is not pre‑bucketed (default 60).
- `--filter`: Optional regex to include specific files (matched against path relative to `--target`).
- `--num_parallel`: Parallelism for per‑subdirectory runs within a bucket.

## How it runs (high level)

1. If `--target` is not already bucketed (`^\d+-\d+$` folder names), files are classified by mtime into interval buckets under `_result/<baseline>/`.
2. A Docker container is started from the instrumented image.
3. For each bucket:
   - The bucket directory is copied into `/root/inputs/<bucket>` in the container.
   - `scripts/acetest_driver.py` is executed inside the container. It:
     - Spawns one process per top‑level subdirectory in the bucket.
     - For each subdirectory, runs `scripts/torch_driver.py` once to execute all `.py` files via `exec()`; exceptions are caught to avoid aborting coverage.
     - Sets `LLVM_PROFILE_FILE` so each subdirectory produces one `.profraw` at `/root/profraw/<bucket>/<subdir>/coverage.profraw`.
     - Merges all `.profraw` to `/root/profraw/<bucket>/merged.profdata`.
   - The merged `.profdata` (and `profile.log` if generated) is copied back to the host under `_result/profdata/<baseline>/<bucket>/`.
4. The container is stopped and removed.

Threading is constrained (OMP/MKL/BLAS/TF env vars) to reduce nondeterminism and resource contention.

## Outputs

- Merged coverage: `_result/profdata/<baseline>/<bucket>/merged.profdata`
- Optional logs: `_result/profdata/<baseline>/<bucket>/profile.log`
- If bucketing was created by the tool: `_result/<baseline>/<bucket>/...` contains copied Python inputs per bucket.

## Drivers

- `scripts/acetest_driver.py` (baseline `acetest`)
  - Discovers top‑level subdirectories under `--inputs-dir`.
  - For each subdirectory, runs `torch_driver.py` once with `LLVM_PROFILE_FILE` pointing to a subdir‑specific `.profraw`.
  - Merges all `.profraw` in the bucket to a single `.profdata`.

- `scripts/torch_driver.py`
  - Executes all `.py` files within a directory (recursively by default) using `compile()` + `exec()`.
  - Catches exceptions and treats `SystemExit` as non‑fatal to keep the batch running.
  - Prints a summary and always returns success so coverage continues.

## Troubleshooting

- Docker image not found: Run `bash build.sh` and ensure `--ver` matches the image tag.
- No Python files found: Check your `--target` path or provide `--filter` if needed.
- Coverage not produced: Ensure your instrumented container is used. The drivers set `LLVM_PROFILE_FILE` for you.
- Performance tuning: Adjust `--num_parallel` and the per‑subdir timeout (`scripts/acetest_driver.py` default is 180s).

## Roadmap

- TensorFlow pipeline is scaffolded with an instrumented Dockerfile. A dedicated collector can be added similar to `TorchCovCollector`.
