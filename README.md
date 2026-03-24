# NeuroRig (WSL2 Optimized)

A lightweight Python diagnostic tool designed for neuroimaging researchers to assess if their hardware can handle intensive MRI processing pipelines (e.g., FreeSurfer, fMRIPrep, FSL, AFNI).

## Purpose
MRI processing is resource-heavy. NeuroRig evaluates:
- **RAM Capacity:** Checks if you have the 16GB-32GB+ required for high-res pipelines.
- **Disk I/O:** Benchmarks read/write speeds (crucial for 4D fMRI datasets).
- **GPU Availability:** Detects NVIDIA CUDA support for accelerated tools like `eddy_cuda` or `FastSurfer`.
- **WSL2 Verification:** Confirms if Windows Subsystem for Linux is correctly seeing your assigned resources.

## Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/neurorig.git
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

   For the extended version with disk I/O benchmarking:
   ```bash
   python neurorig_v2.py
   ```

## Interpreting Results
- **RAM < 16GB:** Stick to basic structural viewing and lightweight preprocessing.
- **Disk < 200 MB/s:** Expect bottlenecks during data-loading; avoid parallel subject processing on this drive.
- **GPU Detected:** You can leverage CUDA-accelerated tools for 10x speed increases.
