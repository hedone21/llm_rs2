import subprocess
import threading
import time
from typing import Optional, Dict

class BenchmarkRunner:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BenchmarkRunner, cls).__new__(cls)
                    cls._instance.process = None
                    cls._instance.log_buffer = []
                    cls._instance.status = "IDLE"
                    cls._instance.current_config = {}
        return cls._instance
    
    def start_run(self, config: Dict) -> bool:
        """
        Starts a new benchmark run if one is not already running.
        config: {
            "backend": "cpu",
            "model": "llama3.2-1b",
            "full_command": "..." # Optional override
        }
        """
        if self.status == "RUNNING":
            return False
            
        self.status = "RUNNING"
        self.log_buffer = [] # Clear logs
        self.current_config = config
        
        # Build command based on config
        # For now, we'll just run a dummy command or the actual script
        # In a real scenario, we'd parse config to build arguments for run_benchmark_suite.py
        
        cmd = ["python3", "-u", "scripts/run_benchmark_suite.py"] # -u for unbuffered output
        if config.get("dry_run"):
            cmd.append("--dry-run")
        if config.get("skip_build"):
            cmd.append("--skip-build")
        if config.get("skip_push"):
            cmd.append("--skip-push")
            
        def run_thread():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd="/home/go/Workspace/llm_rs2"
                )
                
                for line in self.process.stdout:
                    self.log_buffer.append(line)
                    # Keep buffer size manageable? optionally
                    
                self.process.wait()
                if self.process.returncode == 0:
                    self.status = "COMPLETED"
                    self.log_buffer.append("\n[Runner] Process finished successfully.\n")
                else:
                    self.status = "ERROR"
                    self.log_buffer.append(f"\n[Runner] Process finished with error code {self.process.returncode}.\n")
                    
            except Exception as e:
                self.status = "ERROR"
                self.log_buffer.append(f"\n[Runner] Exception: {str(e)}\n")
            finally:
                self.process = None
                # Set status back to IDLE after some time or keep as COMPLETED until acknowledged?
                # For simplicity, let's update status to IDLE after a delay or manual reset.
                # But Requirement says we want to see status. So "COMPLETED" is better.
                # User can reset or start new run which checks if RUNNING.
                pass

        thread = threading.Thread(target=run_thread)
        thread.start()
        return True

    def stop_run(self):
        if self.process and self.status == "RUNNING":
            self.process.terminate()
            self.log_buffer.append("\n[Runner] Terminated by user.\n")
            self.status = "TERMINATED"
            
    def get_status(self):
        return {
            "status": self.status,
            "logs": "".join(self.log_buffer[-100:]), # Return last 100 lines
            "config": self.current_config
        }

runner = BenchmarkRunner()
