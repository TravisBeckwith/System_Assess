# NeuroRig (WSL2 Optimized) 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088305.svg)](https://doi.org/10.5281/zenodo.20088305)
[![CI](https://github.com/TravisBeckwith/NeuroRig/actions/workflows/ci.yml/badge.svg)](https://github.com/TravisBeckwith/NeuroRig/actions/workflows/ci.yml)

A lightweight Python diagnostic tool designed for neuroimaging researchers to assess if their hardware can handle intensive MRI processing pipelines (e.g., FreeSurfer, fMRIPrep, FSL, AFNI).

## Purpose
MRI processing is resource-heavy. NeuroRig evaluates:
- **RAM Capacity:** Checks if you have the 16GB-32GB+ required for high-res pipelines.
- **Disk I/O:** Benchmarks read/write speeds (crucial for 4D fMRI datasets).
- **GPU Availability:** Detects NVIDIA CUDA support for accelerated tools like `eddy_cuda` or `FastSurfer`.
- **WSL2 Verification:** When run inside WSL, confirms whether it's WSL2, reports the RAM visible to the subsystem, and flags the common case where the default 50% host-RAM allocation is bottlenecking you (with a pointer to fix it via `.wslconfig`).

## Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone https://github.com/TravisBeckwith/neurorig.git
   cd neurorig
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the check:**
   ```bash
   python neurorig.py
   ```

   This single script covers CPU, RAM, GPU, disk I/O benchmarking, and (when
   applicable) WSL2 memory-allocation checks — there is no separate v2 file.

## Running Tests
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
Tests run automatically on every push and pull request via GitHub Actions (see badge above).

## Interpreting Results
- **RAM < 16GB:** Stick to basic structural viewing and lightweight preprocessing.
- **Disk < 200 MB/s:** Expect bottlenecks during data-loading; avoid parallel subject processing on this drive.
- **GPU Detected:** You can leverage CUDA-accelerated tools for 10x speed increases.
