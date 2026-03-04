"""
Benchmark Runner — Execute benchmarks on Android via android_profile.py.

Wraps the existing scripts/android_profile.py in a subprocess and tracks
execution state so the frontend can poll for progress.
"""

import subprocess
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROFILE_SCRIPT = _PROJECT_ROOT / "scripts" / "android_profile.py"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "data"

# Android paths (matching run_benchmark_suite.py)
_ANDROID_TMP = "/data/local/tmp"
_LLM_DIR = f"{_ANDROID_TMP}/llm_rs2"
_MODEL_PATH = f"{_LLM_DIR}/models/llama3.2-1b"
_EVAL_DIR = f"{_LLM_DIR}/eval"
_GENERATE_BIN = f"{_ANDROID_TMP}/generate"


class BenchmarkRunner:
    """Singleton-style runner that manages one benchmark at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._process = None
        self._log_lines = []
        self._status = "idle"
        self._params = {}
        self._result_file = None
        self._thread = None

    def is_running(self):
        with self._lock:
            return self._running

    def start(self, backend="cpu", prefill_type="short_len", num_tokens=128):
        """Start a benchmark in a background thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._log_lines = []
            self._status = "starting"
            self._params = {
                "backend": backend,
                "prefill_type": prefill_type,
                "num_tokens": num_tokens,
            }
            self._result_file = None

        self._thread = threading.Thread(
            target=self._run,
            args=(backend, prefill_type, num_tokens),
            daemon=True,
        )
        self._thread.start()

    def _build_device_cmd(self, backend, prefill_type, num_tokens):
        """Build the on-device command string."""
        prompt_file = f"{_EVAL_DIR}/{prefill_type}.txt"
        return (
            f"{_GENERATE_BIN} "
            f"--model-path {_MODEL_PATH} "
            f"--prompt-file {prompt_file} "
            f"--num-tokens {num_tokens} "
            f"-b {backend}"
        )

    def _run(self, backend, prefill_type, num_tokens):
        """Execute android_profile.py in a subprocess."""
        device_cmd = self._build_device_cmd(backend, prefill_type, num_tokens)

        cmd = [
            "python3", str(_PROFILE_SCRIPT),
            "--cmd", device_cmd,
            "--output-dir", str(_RESULTS_DIR),
        ]

        with self._lock:
            self._status = "running"
            self._log_lines.append(f"[Runner] Executing: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(_PROJECT_ROOT),
            )
            with self._lock:
                self._process = proc

            for line in proc.stdout:
                line = line.rstrip("\n")
                with self._lock:
                    self._log_lines.append(line)

            proc.wait()

            with self._lock:
                if proc.returncode == 0:
                    self._status = "completed"
                    self._log_lines.append("[Runner] Benchmark completed successfully.")
                else:
                    self._status = "failed"
                    self._log_lines.append(
                        f"[Runner] Process exited with code {proc.returncode}"
                    )
        except Exception as e:
            with self._lock:
                self._status = "error"
                self._log_lines.append(f"[Runner] Error: {e}")
        finally:
            with self._lock:
                self._running = False
                self._process = None

    def get_status(self):
        """Return current status and log lines."""
        with self._lock:
            return {
                "status": self._status,
                "running": self._running,
                "params": self._params,
                "log": list(self._log_lines),
                "log_count": len(self._log_lines),
            }
