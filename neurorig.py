import psutil
import platform
import subprocess
import shutil
import os
import time


def get_size(num_bytes, suffix="B"):
    """Scale bytes to its proper format (e.g., MB, GB)."""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if num_bytes < factor:
            return f"{num_bytes:.2f}{unit}{suffix}"
        num_bytes /= factor
    return f"{num_bytes:.2f}E{suffix}"


def check_gpu():
    """Check for NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            return "NVIDIA GPU found, but nvidia-smi failed."
    except FileNotFoundError:
        return "No NVIDIA GPU detected (or nvidia-smi not in PATH)."
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out — GPU status unknown."
    except Exception as e:
        return f"Could not determine GPU status ({e})."


def check_wsl2():
    """Detect whether we're running under WSL2 and flag common memory-limiting gotchas."""
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
    except (FileNotFoundError, PermissionError):
        return None  # Not Linux, or /proc unavailable — not WSL

    if "microsoft" not in version_info and "wsl" not in version_info:
        return None  # Native Linux, not WSL

    is_wsl2 = "wsl2" in version_info or "microsoft-standard-wsl2" in version_info

    lines = [f"Running under {'WSL2' if is_wsl2 else 'WSL (version undetermined)'}."]

    # WSL2 defaults to 50% of host RAM (or a .wslconfig override). Flag anything under
    # 8GB as likely to bottleneck real pipelines and suggest a .wslconfig fix.
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if total_ram_gb < 8:
        lines.append(
            f"⚠️  Only {total_ram_gb:.1f}GB RAM visible to WSL2 — this is likely capped by "
            "the default 50% host allocation. Consider raising 'memory=' in %UserProfile%\\.wslconfig "
            "and restarting WSL ('wsl --shutdown')."
        )
    else:
        lines.append(f"RAM visible to WSL2: {total_ram_gb:.1f}GB.")

    return "\n".join(lines)


def test_disk_speed(file_size_mb=500):
    """Benchmark disk read/write speeds. Returns (write_mbps, read_mbps) or (None, None) on failure."""
    chunk_size = 1024 * 1024  # 1 MB chunk
    chunks = file_size_mb

    temp_dir = os.getcwd()
    test_file = os.path.join(temp_dir, "neurorig_io_test_file.tmp")
    data = b'\x00' * chunk_size

    try:
        # --- Write Test ---
        start_time = time.time()
        with open(test_file, 'wb') as f:
            for _ in range(chunks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())  # Force write to physical disk
        write_time = time.time() - start_time
        write_speed = file_size_mb / write_time if write_time > 0 else float('inf')

        # --- Read Test ---
        # Note: OS caching might make this artificially fast,
        # but it gives a general ballpark for I/O capability.
        start_time = time.time()
        with open(test_file, 'rb') as f:
            while f.read(chunk_size):
                pass
        read_time = time.time() - start_time
        read_speed = file_size_mb / read_time if read_time > 0 else float('inf')

        return write_speed, read_speed

    except (OSError, IOError) as e:
        print(f"⚠️  Disk benchmark skipped: {e}")
        return None, None
    finally:
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
            except OSError:
                pass


def run_diagnostics():
    print("=" * 50)
    print("🧠 NEURORIG — MRI HARDWARE DIAGNOSTICS 🧠")
    print("=" * 50)

    # 1. CPU Info
    print("\n--- Processor (CPU) ---")
    print(f"Processor: {platform.processor()}")
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Physical cores: {physical_cores if physical_cores is not None else 'Unknown'}")
    print(f"Total threads: {logical_cores if logical_cores is not None else 'Unknown'}")

    # 2. RAM Info
    print("\n--- Memory (RAM) ---")
    svmem = psutil.virtual_memory()
    print(f"Total RAM: {get_size(svmem.total)}")
    print(f"Available RAM: {get_size(svmem.available)}")

    # 3. GPU Info
    print("\n--- Graphics Processing Unit (GPU) ---")
    print(check_gpu())

    # 4. WSL2 Info (only prints if actually running under WSL)
    wsl_status = check_wsl2()
    if wsl_status:
        print("\n--- WSL2 Environment ---")
        print(wsl_status)

    # 5. Storage & I/O Info
    print("\n--- Storage (Disk Space & Speed) ---")
    total, used, free = shutil.disk_usage(os.getcwd())
    print(f"Total Space on Current Drive: {get_size(total)}")
    print(f"Free Space on Current Drive: {get_size(free)}")

    print("\nRunning Disk I/O Benchmark (Writing/Reading 500MB)...")
    write_mbps, read_mbps = test_disk_speed()
    if write_mbps is not None:
        print(f"Sequential Write Speed: {write_mbps:.2f} MB/s")
        print(f"Sequential Read Speed:  {read_mbps:.2f} MB/s")
    else:
        print("Disk benchmark could not complete (see warning above).")

    print("\n" + "=" * 50)
    print("📊 NEURORIG CAPABILITY ASSESSMENT 📊")
    print("=" * 50)

    # Heuristics
    ram_gb = svmem.total / (1024 ** 3)
    cores = physical_cores if physical_cores is not None else logical_cores

    print(f"RAM Tier:   {'✅ High' if ram_gb >= 30 else '⚠️ Medium' if ram_gb >= 15 else '❌ Low'}")

    if cores is None:
        print("CPU Tier:   ❓ Unknown (could not determine core count)")
    else:
        print(f"CPU Tier:   {'✅ High' if cores >= 8 else '⚠️ Medium' if cores >= 4 else '❌ Low'}")

    if write_mbps is None:
        print("Drive Tier: ❓ Unknown (benchmark failed)")
    else:
        print(f"Drive Tier: {'✅ High (NVMe/Fast SSD)' if write_mbps > 1000 else '⚠️ Medium (SATA SSD)' if write_mbps > 300 else '❌ Low (HDD/Slow)'}")


if __name__ == "__main__":
    try:
        run_diagnostics()
    except KeyboardInterrupt:
        print("\nDiagnostics cancelled.")
    except Exception as e:
        print(f"\n❌ NeuroRig hit an unexpected error: {e}")
        print("Please report this at the project's GitHub issues page.")
