import psutil
import platform
import subprocess
import shutil
import os
import time
import tempfile

def get_size(bytes, suffix="B"):
    """Scale bytes to its proper format (e.g., MB, GB)."""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def check_gpu():
    """Check for NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "NVIDIA GPU found, but nvidia-smi failed."
    except FileNotFoundError:
        return "No NVIDIA GPU detected (or nvidia-smi not in PATH)."

def test_disk_speed(file_size_mb=500):
    """Benchmark disk read/write speeds."""
    chunk_size = 1024 * 1024  # 1 MB chunk
    chunks = file_size_mb
    
    # Use a temporary file in the current working directory 
    # (to test the specific drive where your data/processing will happen)
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
            os.fsync(f.fileno()) # Force write to physical disk
        write_time = time.time() - start_time
        write_speed = file_size_mb / write_time
        
        # --- Read Test ---
        # Note: OS caching might make this artificially fast, 
        # but it gives a general ballpark for I/O capability.
        start_time = time.time()
        with open(test_file, 'rb') as f:
            while f.read(chunk_size):
                pass
        read_time = time.time() - start_time
        read_speed = file_size_mb / read_time
        
        return write_speed, read_speed
        
    finally:
        # Clean up the test file
        if os.path.exists(test_file):
            os.remove(test_file)

def run_diagnostics():
    print("="*50)
    print("🧠 NEURORIG — MRI HARDWARE DIAGNOSTICS 🧠")
    print("="*50)

    # 1. CPU Info
    print("\n--- Processor (CPU) ---")
    print(f"Processor: {platform.processor()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total threads: {psutil.cpu_count(logical=True)}")

    # 2. RAM Info
    print("\n--- Memory (RAM) ---")
    svmem = psutil.virtual_memory()
    print(f"Total RAM: {get_size(svmem.total)}")
    print(f"Available RAM: {get_size(svmem.available)}")
    
    # 3. GPU Info
    print("\n--- Graphics Processing Unit (GPU) ---")
    print(check_gpu())

    # 4. Storage & I/O Info
    print("\n--- Storage (Disk Space & Speed) ---")
    total, used, free = shutil.disk_usage(os.getcwd())
    print(f"Total Space on Current Drive: {get_size(total)}")
    print(f"Free Space on Current Drive: {get_size(free)}")
    
    print("\nRunning Disk I/O Benchmark (Writing/Reading 500MB)...")
    write_mbps, read_mbps = test_disk_speed()
    print(f"Sequential Write Speed: {write_mbps:.2f} MB/s")
    print(f"Sequential Read Speed:  {read_mbps:.2f} MB/s")
    
    print("\n" + "="*50)
    print("📊 NEURORIG CAPABILITY ASSESSMENT 📊")
    print("="*50)
    
    # Heuristics
    ram_gb = svmem.total / (1024**3)
    cores = psutil.cpu_count(logical=False)
    
    print(f"RAM Tier:   {'✅ High' if ram_gb >= 30 else '⚠️ Medium' if ram_gb >= 15 else '❌ Low'}")
    print(f"CPU Tier:   {'✅ High' if cores >= 8 else '⚠️ Medium' if cores >= 4 else '❌ Low'}")
    print(f"Drive Tier: {'✅ High (NVMe/Fast SSD)' if write_mbps > 1000 else '⚠️ Medium (SATA SSD)' if write_mbps > 300 else '❌ Low (HDD/Slow)'}")

if __name__ == "__main__":
    run_diagnostics()
