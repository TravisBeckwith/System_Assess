"""
Test suite for neurorig.py

Run with: pytest -v
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurorig  # noqa: E402


# --------------------------------------------------------------------------
# get_size
# --------------------------------------------------------------------------

class TestGetSize:
    def test_bytes(self):
        assert neurorig.get_size(500) == "500.00B"

    def test_kilobytes(self):
        assert neurorig.get_size(2048) == "2.00KB"

    def test_gigabytes(self):
        assert neurorig.get_size(2 * 1024 ** 3) == "2.00GB"

    def test_zero(self):
        assert neurorig.get_size(0) == "0.00B"

    def test_does_not_mutate_builtin_bytes(self):
        # Regression test: the original code shadowed the `bytes` builtin
        # by naming a parameter `bytes`, which is fragile. Confirm the
        # builtin is untouched after calling get_size.
        neurorig.get_size(123)
        assert bytes is not None
        assert callable(bytes)


# --------------------------------------------------------------------------
# check_gpu
# --------------------------------------------------------------------------

class TestCheckGpu:
    def test_gpu_found(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="Tesla T4, 16384 MiB, 15000 MiB\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = neurorig.check_gpu()
        assert "Tesla T4" in result

    def test_no_nvidia_smi_binary(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = neurorig.check_gpu()
        assert "No NVIDIA GPU detected" in result

    def test_nvidia_smi_nonzero_exit(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="error")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = neurorig.check_gpu()
        assert "failed" in result.lower()

    def test_nvidia_smi_timeout(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = neurorig.check_gpu()
        assert "timed out" in result.lower()

    def test_unexpected_exception_does_not_crash(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = neurorig.check_gpu()
        assert isinstance(result, str)


# --------------------------------------------------------------------------
# check_wsl2
# --------------------------------------------------------------------------

class TestCheckWsl2:
    @staticmethod
    def _patch_proc_version(monkeypatch, tmp_path, content):
        """Redirect only the /proc/version read to fake content, leaving
        all other open() calls (e.g. within the fake file itself) untouched."""
        fake_proc_version = tmp_path / "version"
        fake_proc_version.write_text(content)
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if path == "/proc/version":
                return real_open(fake_proc_version, mode, *args, **kwargs)
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

    def test_not_wsl_returns_none(self, monkeypatch, tmp_path):
        self._patch_proc_version(monkeypatch, tmp_path, "Linux version 6.8.0-generic (Ubuntu)")
        assert neurorig.check_wsl2() is None

    def test_wsl2_detected(self, monkeypatch, tmp_path):
        self._patch_proc_version(
            monkeypatch, tmp_path,
            "Linux version 5.15.153.1-microsoft-standard-WSL2 (root@...)"
        )
        result = neurorig.check_wsl2()
        assert result is not None
        assert "WSL2" in result

    def test_low_ram_warning(self, monkeypatch, tmp_path):
        self._patch_proc_version(
            monkeypatch, tmp_path,
            "Linux version 5.15.153.1-microsoft-standard-WSL2 (root@...)"
        )

        class FakeMem:
            total = 4 * 1024 ** 3  # 4GB, below the 8GB warning threshold

        monkeypatch.setattr(neurorig.psutil, "virtual_memory", lambda: FakeMem())

        result = neurorig.check_wsl2()
        assert "wslconfig" in result

    def test_no_proc_version_file(self, monkeypatch):
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if path == "/proc/version":
                raise FileNotFoundError(path)
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        assert neurorig.check_wsl2() is None


# --------------------------------------------------------------------------
# test_disk_speed
# --------------------------------------------------------------------------

class TestDiskSpeed:
    def test_returns_positive_speeds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_speed, read_speed = neurorig.test_disk_speed(file_size_mb=5)
        assert write_speed is not None
        assert read_speed is not None
        assert write_speed > 0
        assert read_speed > 0

    def test_cleans_up_temp_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        neurorig.test_disk_speed(file_size_mb=2)
        leftover = tmp_path / "neurorig_io_test_file.tmp"
        assert not leftover.exists()

    def test_handles_unwritable_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def fake_open(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("builtins.open", fake_open)
        write_speed, read_speed = neurorig.test_disk_speed(file_size_mb=2)
        assert write_speed is None
        assert read_speed is None

    def test_no_division_by_zero_on_instant_write(self, tmp_path, monkeypatch):
        # Regression test: a write/read that completes in 0.0s (e.g. on a
        # ramdisk) must not raise ZeroDivisionError.
        monkeypatch.chdir(tmp_path)

        times = iter([100.0, 100.0, 200.0, 200.0])  # start==end for both phases
        monkeypatch.setattr(neurorig.time, "time", lambda: next(times))

        write_speed, read_speed = neurorig.test_disk_speed(file_size_mb=1)
        assert write_speed == float("inf")
        assert read_speed == float("inf")


# --------------------------------------------------------------------------
# run_diagnostics smoke test
# --------------------------------------------------------------------------

class TestRunDiagnostics:
    def test_runs_without_raising(self, capsys, tmp_path, monkeypatch):
        """End-to-end smoke test: the full report should run without
        crashing, even in a minimal/sandboxed CI environment."""
        monkeypatch.chdir(tmp_path)
        neurorig.run_diagnostics()
        captured = capsys.readouterr()
        assert "NEURORIG" in captured.out
        assert "CAPABILITY ASSESSMENT" in captured.out

    def test_survives_none_physical_cores(self, capsys, tmp_path, monkeypatch):
        # Regression test: psutil.cpu_count(logical=False) can return None
        # on some platforms/containers; this must not crash the tier report.
        monkeypatch.chdir(tmp_path)

        real_cpu_count = neurorig.psutil.cpu_count

        def fake_cpu_count(logical=True):
            if not logical:
                return None
            return real_cpu_count(logical=True)

        monkeypatch.setattr(neurorig.psutil, "cpu_count", fake_cpu_count)
        neurorig.run_diagnostics()
        captured = capsys.readouterr()
        assert "CPU Tier" in captured.out
