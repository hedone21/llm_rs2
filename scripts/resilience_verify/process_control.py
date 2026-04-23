"""Local subprocess and remote (SSH) process control for resilience verify.

Local path (BackgroundProcess / run_foreground) — unchanged from Phase 1~3.
Remote path (SshBackgroundProcess / run_foreground_ssh) — added in Phase 4.

SSH design notes:
- We shell out directly to `ssh -p <port> -o BatchMode=yes user@host '<cmd>'`
  rather than using paramiko; this matches the rest of the scripts/ codebase
  and avoids another dependency.
- Remote background processes are started via `nohup ... >log 2>&1 & echo $!`
  and tracked via an on-device pidfile. The stdbuf wrapper forces line-buffered
  stdout/stderr so logs flush promptly.
- The openssh client emits a stderr WARNING line about post-quantum KEX on this
  host. We redirect *our* ssh-call stderr into dedicated `.ssh.stderr` logs so
  the warning never contaminates the remote log files we parse downstream.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ── Local subprocess (Phase 1) ──────────────────────────────────────────

class BackgroundProcess:
    """Context manager that launches a local subprocess and captures logs.

    Usage:
        with BackgroundProcess([...], stdout_path=..., stderr_path=...) as bg:
            ...
        # on exit: SIGTERM → SIGKILL sequence
    """

    def __init__(
        self,
        cmd: Sequence[str],
        stdout_path: Path,
        stderr_path: Path,
        env: Optional[dict] = None,
        cwd: Optional[Path] = None,
        term_timeout_s: float = 3.0,
    ):
        self.cmd: List[str] = list(cmd)
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self.env = env
        self.cwd = cwd
        self.term_timeout_s = term_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_f = None
        self._stderr_f = None

    def __enter__(self) -> "BackgroundProcess":
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_f = open(self.stdout_path, "wb")
        self._stderr_f = open(self.stderr_path, "wb")
        self._proc = subprocess.Popen(
            self.cmd,
            stdout=self._stdout_f,
            stderr=self._stderr_f,
            env=self.env,
            cwd=str(self.cwd) if self.cwd else None,
        )
        return self

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    def poll(self) -> Optional[int]:
        return self._proc.poll() if self._proc else None

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if not self._proc:
            return None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def terminate(self):
        if not self._proc:
            return
        if self._proc.poll() is not None:
            return
        try:
            self._proc.terminate()
        except ProcessLookupError:
            return
        try:
            self._proc.wait(timeout=self.term_timeout_s)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
            except Exception:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        for f in (self._stdout_f, self._stderr_f):
            try:
                if f:
                    f.close()
            except Exception:
                pass
        return False


def run_foreground(
    cmd: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    timeout_s: float = 60.0,
) -> tuple[int, bool]:
    """Run a subprocess in foreground, capturing logs."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
        proc = subprocess.Popen(
            list(cmd),
            stdout=out_f,
            stderr=err_f,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            rc = -1
    return rc, timed_out


def pkill_pattern(pattern: str) -> None:
    """Best-effort pkill by name pattern (local host only)."""
    try:
        subprocess.run(
            ["pkill", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        pass


def remove_socket_files(patterns: Sequence[str]) -> None:
    """Delete stray unix socket files matching glob patterns."""
    from glob import glob

    for pat in patterns:
        for path in glob(pat):
            try:
                os.unlink(path)
            except OSError:
                pass


# ── SSH remote (Phase 4) ─────────────────────────────────────────────────


class SshRemote:
    """Lightweight SSH helper around the openssh client.

    Holds (user, host, port, identity_file) and exposes:
    - exec(cmd, timeout, err_log, env_vars) → (rc, stdout_text, stderr_text)
    - push(local, remote, retries)
    - pull(remote, local, retries)
    - exists(remote_path)
    - mkdir(remote_path)
    - rm(remote_path)

    The `env_vars` mapping is prefixed verbatim inline on the remote command
    (e.g. `LD_LIBRARY_PATH=/usr/local/cuda/lib64 /home/nvidia/llm_rs2/generate`).
    """

    def __init__(
        self,
        host: str,
        user: str,
        port: int = 22,
        identity_file: str = "",
        extra_options: Optional[List[str]] = None,
    ) -> None:
        if not host or not user:
            raise ValueError("SshRemote requires non-empty host and user")
        self.host = host
        self.user = user
        self.port = port
        self.identity_file = identity_file
        self.extra_options: List[str] = list(extra_options or [])

    def _ssh_base(self) -> List[str]:
        args = [
            "ssh",
            "-p", str(self.port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]
        if self.identity_file:
            args.extend(["-i", self.identity_file])
        args.extend(self.extra_options)
        args.append(f"{self.user}@{self.host}")
        return args

    def _scp_base(self) -> List[str]:
        args = [
            "scp",
            "-P", str(self.port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]
        if self.identity_file:
            args.extend(["-i", self.identity_file])
        args.extend(self.extra_options)
        return args

    @staticmethod
    def _inline_env(env_vars: Optional[Dict[str, str]]) -> str:
        if not env_vars:
            return ""
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items()) + " "

    def exec(
        self,
        remote_cmd: str,
        timeout: Optional[float] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> tuple[int, str, str]:
        """Run `remote_cmd` synchronously via ssh, return (rc, stdout, stderr)."""
        prefix = self._inline_env(env_vars)
        full = self._ssh_base() + [f"{prefix}{remote_cmd}"]
        try:
            r = subprocess.run(
                full, capture_output=True, text=True, timeout=timeout
            )
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"ssh timed out after {timeout}s"

    def push(
        self,
        local_path: Path,
        remote_path: str,
        retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        local = Path(local_path)
        last_err = ""
        for attempt in range(retries + 1):
            cmd = self._scp_base()
            if local.is_dir():
                cmd.append("-r")
            cmd.extend([str(local), f"{self.user}@{self.host}:{remote_path}"])
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                if r.returncode == 0:
                    return
                last_err = r.stderr.strip() or r.stdout.strip() or f"rc={r.returncode}"
            except subprocess.TimeoutExpired:
                last_err = f"scp push timed out after {timeout}s"
            if attempt < retries:
                time.sleep(2.0)
        raise RuntimeError(
            f"scp push failed after {retries+1} attempts: "
            f"{local} -> {self.user}@{self.host}:{remote_path} — {last_err}"
        )

    def pull(
        self,
        remote_path: str,
        local_path: Path,
        retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        last_err = ""
        for attempt in range(retries + 1):
            cmd = self._scp_base() + [
                f"{self.user}@{self.host}:{remote_path}",
                str(local),
            ]
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                if r.returncode == 0:
                    return
                last_err = r.stderr.strip() or r.stdout.strip() or f"rc={r.returncode}"
            except subprocess.TimeoutExpired:
                last_err = f"scp pull timed out after {timeout}s"
            if attempt < retries:
                time.sleep(2.0)
        # Soft-fail: create empty local file so downstream parsers don't crash.
        if not local.exists():
            local.write_bytes(b"")
        raise RuntimeError(
            f"scp pull failed after {retries+1} attempts: "
            f"{self.user}@{self.host}:{remote_path} -> {local} — {last_err}"
        )

    def exists(self, remote_path: str) -> bool:
        rc, _, _ = self.exec(f"test -e {shlex.quote(remote_path)}", timeout=15.0)
        return rc == 0

    def mkdir(self, remote_path: str) -> None:
        self.exec(f"mkdir -p {shlex.quote(remote_path)}", timeout=15.0)

    def rm(self, remote_path: str) -> None:
        self.exec(f"rm -rf {shlex.quote(remote_path)}", timeout=15.0)

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        self.exec(f"chmod {mode} {shlex.quote(remote_path)}", timeout=15.0)


def ssh_remote_from_dict(conn: Dict[str, Any]) -> SshRemote:
    """Build an SshRemote from the `connection` sub-table of devices.toml."""
    return SshRemote(
        host=conn["host"],
        user=conn["user"],
        port=int(conn.get("port", 22)),
        identity_file=conn.get("identity_file", ""),
        extra_options=list(conn.get("ssh_options", [])),
    )


class SshBackgroundProcess:
    """Context-managed remote background process launched via ssh.

    Semantics:
    - __enter__ runs a blocking `ssh` one-shot that kicks off the command via
      `nohup ... & echo $! > pidfile` on the remote side, then exits.
    - The ssh call itself returns quickly (the remote shell is detached from
      our local ssh session via `</dev/null &`). After a short wait we pull
      the pidfile to discover the remote PID.
    - __exit__ issues SIGTERM, waits, then SIGKILL on the remote PID; removes
      the pidfile.

    `stdbuf -oL -eL` is prepended so remote log files flush line-by-line.
    """

    def __init__(
        self,
        remote: SshRemote,
        cmd: Sequence[str],
        remote_log: str,
        remote_pidfile: str,
        ssh_stderr_log: Path,
        env_vars: Optional[Dict[str, str]] = None,
        startup_wait_s: float = 0.5,
        term_wait_s: float = 1.0,
    ):
        self.remote = remote
        self.cmd: List[str] = list(cmd)
        self.remote_log = remote_log
        self.remote_pidfile = remote_pidfile
        self.ssh_stderr_log = Path(ssh_stderr_log)
        self.env_vars = env_vars or {}
        self.startup_wait_s = startup_wait_s
        self.term_wait_s = term_wait_s
        self.remote_pid: Optional[int] = None

    def _quote_cmd(self) -> str:
        # Remote-side shell quoting.
        return " ".join(shlex.quote(a) for a in self.cmd)

    def __enter__(self) -> "SshBackgroundProcess":
        env_prefix = SshRemote._inline_env(self.env_vars)
        cmd_str = self._quote_cmd()
        remote_log = shlex.quote(self.remote_log)
        remote_pidfile = shlex.quote(self.remote_pidfile)
        # nohup + stdbuf + background + echo PID.
        # Use sh -c so the whole thing is one shell, otherwise the $! is wrong.
        # NOTE: we redirect BOTH stdout and stderr to the log (2>&1). Remote
        # processes should get their own file; ssh-call stderr goes separately.
        remote_script = (
            f"{env_prefix}nohup stdbuf -oL -eL {cmd_str} "
            f"> {remote_log} 2>&1 < /dev/null & "
            f"echo $! > {remote_pidfile}; "
            "disown || true; "
            "exit 0"
        )
        self.ssh_stderr_log.parent.mkdir(parents=True, exist_ok=True)

        with open(self.ssh_stderr_log, "ab") as err_f:
            full = self.remote._ssh_base() + [f"sh -c {shlex.quote(remote_script)}"]
            r = subprocess.run(
                full,
                stdout=subprocess.DEVNULL,
                stderr=err_f,
                timeout=30.0,
            )
        if r.returncode != 0:
            raise RuntimeError(
                f"SshBackgroundProcess failed to launch (ssh rc={r.returncode}); "
                f"see {self.ssh_stderr_log}"
            )

        # Give nohup a moment to fork, then read pidfile.
        time.sleep(self.startup_wait_s)
        rc, stdout, _stderr = self.remote.exec(
            f"cat {shlex.quote(self.remote_pidfile)} 2>/dev/null || true",
            timeout=10.0,
        )
        text = stdout.strip()
        if rc == 0 and text.isdigit():
            self.remote_pid = int(text)
        return self

    def poll_alive(self) -> bool:
        if self.remote_pid is None:
            return False
        rc, _stdout, _stderr = self.remote.exec(
            f"kill -0 {self.remote_pid} 2>/dev/null",
            timeout=10.0,
        )
        return rc == 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.remote_pid is None:
            return False
        pid = self.remote_pid
        term_cmd = (
            f"kill -TERM {pid} 2>/dev/null; "
            f"sleep {self.term_wait_s}; "
            f"kill -KILL {pid} 2>/dev/null; "
            f"rm -f {shlex.quote(self.remote_pidfile)}"
        )
        try:
            self.remote.exec(term_cmd, timeout=15.0)
        except Exception:
            pass
        return False


def run_foreground_ssh(
    remote: SshRemote,
    cmd: Sequence[str],
    local_stdout_path: Path,
    local_stderr_path: Path,
    ssh_stderr_path: Path,
    remote_log_stdout: str,
    remote_log_stderr: str,
    env_vars: Optional[Dict[str, str]] = None,
    timeout_s: float = 600.0,
) -> tuple[int, bool]:
    """Run `cmd` synchronously on the remote host; capture stdout/stderr to
    remote files; on completion, scp-pull them to local paths.

    `ssh_stderr_path` captures the openssh client's own stderr (the PQ KEX
    WARNING), kept separate from the remote process stderr.
    """
    env_prefix = SshRemote._inline_env(env_vars)
    cmd_str = " ".join(shlex.quote(a) for a in cmd)
    remote_log_stdout_q = shlex.quote(remote_log_stdout)
    remote_log_stderr_q = shlex.quote(remote_log_stderr)

    # Run foreground remotely with line-buffered output:
    remote_script = (
        f"{env_prefix}stdbuf -oL -eL {cmd_str} "
        f"> {remote_log_stdout_q} 2> {remote_log_stderr_q}"
    )

    ssh_stderr_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    rc = -1
    with open(ssh_stderr_path, "ab") as ssh_err_f:
        full = remote._ssh_base() + [f"sh -c {shlex.quote(remote_script)}"]
        try:
            r = subprocess.run(
                full,
                stdout=subprocess.DEVNULL,
                stderr=ssh_err_f,
                timeout=timeout_s,
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -1

    # Pull logs regardless of rc (even if timeout we may still have partial logs).
    try:
        remote.pull(remote_log_stdout, local_stdout_path, retries=2, timeout=60.0)
    except Exception as e:
        # Best-effort: write the scp error into the stderr file.
        try:
            with open(local_stderr_path, "ab") as f:
                f.write(f"\n[run_foreground_ssh] stdout pull failed: {e!r}\n".encode())
        except Exception:
            pass
    try:
        remote.pull(remote_log_stderr, local_stderr_path, retries=2, timeout=60.0)
    except Exception as e:
        try:
            with open(local_stderr_path, "ab") as f:
                f.write(f"\n[run_foreground_ssh] stderr pull failed: {e!r}\n".encode())
        except Exception:
            pass

    return rc, timed_out


# ── Phase 5 (adb, Android) ──────────────────────────────────────────────


class AdbRemote:
    """Lightweight helper around `adb`.

    Mirrors SshRemote's shape but targets an Android device via `adb -s <serial>`.
    - exec(cmd, timeout, env_vars) → (rc, stdout, stderr)
    - push(local, remote) / pull(remote, local) via `adb push|pull`
    - mkdir / rm / chmod / exists

    Environment variables are prefixed inline on the remote shell command,
    same as the ssh path (e.g. `LD_LIBRARY_PATH=/data/local/tmp ...`).
    """

    def __init__(self, serial: str = "") -> None:
        self.serial = serial

    def _adb_base(self) -> List[str]:
        if self.serial:
            return ["adb", "-s", self.serial]
        return ["adb"]

    @staticmethod
    def _inline_env(env_vars: Optional[Dict[str, str]]) -> str:
        if not env_vars:
            return ""
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items()) + " "

    def exec(
        self,
        remote_cmd: str,
        timeout: Optional[float] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> tuple[int, str, str]:
        prefix = self._inline_env(env_vars)
        full = self._adb_base() + ["shell", f"{prefix}{remote_cmd}"]
        try:
            r = subprocess.run(
                full, capture_output=True, text=True, timeout=timeout
            )
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"adb timed out after {timeout}s"

    def push(
        self,
        local_path: Path,
        remote_path: str,
        retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        local = Path(local_path)
        last_err = ""
        for attempt in range(retries + 1):
            cmd = self._adb_base() + ["push", str(local), remote_path]
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                if r.returncode == 0:
                    return
                last_err = r.stderr.strip() or r.stdout.strip() or f"rc={r.returncode}"
            except subprocess.TimeoutExpired:
                last_err = f"adb push timed out after {timeout}s"
            if attempt < retries:
                time.sleep(2.0)
        raise RuntimeError(
            f"adb push failed after {retries+1} attempts: "
            f"{local} -> {remote_path} — {last_err}"
        )

    def pull(
        self,
        remote_path: str,
        local_path: Path,
        retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        last_err = ""
        for attempt in range(retries + 1):
            cmd = self._adb_base() + ["pull", remote_path, str(local)]
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                if r.returncode == 0:
                    return
                last_err = r.stderr.strip() or r.stdout.strip() or f"rc={r.returncode}"
            except subprocess.TimeoutExpired:
                last_err = f"adb pull timed out after {timeout}s"
            if attempt < retries:
                time.sleep(2.0)
        # Soft-fail: create empty local file so downstream parsers don't crash.
        if not local.exists():
            local.write_bytes(b"")
        raise RuntimeError(
            f"adb pull failed after {retries+1} attempts: "
            f"{remote_path} -> {local} — {last_err}"
        )

    def exists(self, remote_path: str) -> bool:
        rc, _, _ = self.exec(f"test -e {shlex.quote(remote_path)} && echo Y || echo N", timeout=15.0)
        # adb shell always returns 0 for `echo`, so read stdout.
        # Simpler: run `ls` on the path.
        rc2, stdout, _ = self.exec(f"ls {shlex.quote(remote_path)} 2>/dev/null", timeout=15.0)
        return bool(stdout.strip())

    def mkdir(self, remote_path: str) -> None:
        self.exec(f"mkdir -p {shlex.quote(remote_path)}", timeout=15.0)

    def rm(self, remote_path: str) -> None:
        self.exec(f"rm -rf {shlex.quote(remote_path)}", timeout=15.0)

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        self.exec(f"chmod {mode} {shlex.quote(remote_path)}", timeout=15.0)


def adb_remote_from_dict(conn: Dict[str, Any]) -> "AdbRemote":
    """Build an AdbRemote from the `connection` sub-table of devices.toml.

    If `serial` is empty and multiple devices are attached, adb will fail —
    caller should pre-resolve via `adb devices`.
    """
    serial = str(conn.get("serial", "") or "")
    return AdbRemote(serial=serial)


def resolve_adb_serial(preferred: str = "") -> str:
    """Return a concrete adb serial.

    - If `preferred` is given, return it verbatim.
    - Otherwise, query `adb devices` and return the single attached device's
      serial. If zero or >1 devices, raise RuntimeError.
    """
    if preferred:
        return preferred
    r = subprocess.run(
        ["adb", "devices"], capture_output=True, text=True, timeout=10.0
    )
    if r.returncode != 0:
        raise RuntimeError(f"adb devices failed: {r.stderr!r}")
    lines = [ln.strip() for ln in r.stdout.splitlines()[1:] if ln.strip()]
    serials = [ln.split()[0] for ln in lines if "device" in ln.split()]
    if len(serials) == 0:
        raise RuntimeError("No adb devices attached")
    if len(serials) > 1:
        raise RuntimeError(
            f"Multiple adb devices attached ({serials}); set connection.serial"
        )
    return serials[0]


class AdbBackgroundProcess:
    """Context-managed remote background process launched via `adb shell`.

    Semantics mirror SshBackgroundProcess:
    - __enter__ fires an `adb shell` one-shot that starts the binary via
      `nohup ... & echo $! > pidfile` and returns immediately.
    - PID is read from the pidfile shortly after launch.
    - __exit__ issues SIGTERM then SIGKILL on the remote PID.

    `stdbuf -oL -eL` is prepended so log files flush line-by-line.
    """

    def __init__(
        self,
        remote: AdbRemote,
        cmd: Sequence[str],
        remote_log: str,
        remote_pidfile: str,
        adb_stderr_log: Path,
        env_vars: Optional[Dict[str, str]] = None,
        startup_wait_s: float = 0.5,
        term_wait_s: float = 1.0,
    ):
        self.remote = remote
        self.cmd: List[str] = list(cmd)
        self.remote_log = remote_log
        self.remote_pidfile = remote_pidfile
        self.adb_stderr_log = Path(adb_stderr_log)
        self.env_vars = env_vars or {}
        self.startup_wait_s = startup_wait_s
        self.term_wait_s = term_wait_s
        self.remote_pid: Optional[int] = None

    def _quote_cmd(self) -> str:
        return " ".join(shlex.quote(a) for a in self.cmd)

    def __enter__(self) -> "AdbBackgroundProcess":
        env_prefix = AdbRemote._inline_env(self.env_vars)
        cmd_str = self._quote_cmd()
        remote_log = shlex.quote(self.remote_log)
        remote_pidfile = shlex.quote(self.remote_pidfile)
        # On Android, `stdbuf` is NOT always available in /system/bin. Use
        # `sh -c` and accept the possible buffering — mock_manager println!()
        # flushes each line anyway.
        remote_script = (
            f"{env_prefix}nohup {cmd_str} "
            f"> {remote_log} 2>&1 < /dev/null & "
            f"echo $! > {remote_pidfile}; "
            "disown 2>/dev/null || true; "
            "exit 0"
        )
        self.adb_stderr_log.parent.mkdir(parents=True, exist_ok=True)

        with open(self.adb_stderr_log, "ab") as err_f:
            full = self.remote._adb_base() + ["shell", f"sh -c {shlex.quote(remote_script)}"]
            r = subprocess.run(
                full,
                stdout=subprocess.DEVNULL,
                stderr=err_f,
                timeout=30.0,
            )
        if r.returncode != 0:
            raise RuntimeError(
                f"AdbBackgroundProcess failed to launch (adb rc={r.returncode}); "
                f"see {self.adb_stderr_log}"
            )

        time.sleep(self.startup_wait_s)
        rc, stdout, _stderr = self.remote.exec(
            f"cat {shlex.quote(self.remote_pidfile)} 2>/dev/null || true",
            timeout=10.0,
        )
        text = stdout.strip()
        if rc == 0 and text.isdigit():
            self.remote_pid = int(text)
        return self

    def poll_alive(self) -> bool:
        if self.remote_pid is None:
            return False
        rc, _stdout, _stderr = self.remote.exec(
            f"kill -0 {self.remote_pid} 2>/dev/null && echo A || echo D",
            timeout=10.0,
        )
        return "A" in (_stdout or "")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.remote_pid is None:
            # Best effort pkill by binary name fragment (pidfile may be missing
            # if the launch racily died).
            try:
                self.remote.exec(
                    f"rm -f {shlex.quote(self.remote_pidfile)} 2>/dev/null; true",
                    timeout=10.0,
                )
            except Exception:
                pass
            return False
        pid = self.remote_pid
        term_cmd = (
            f"kill -TERM {pid} 2>/dev/null; "
            f"sleep {self.term_wait_s}; "
            f"kill -KILL {pid} 2>/dev/null; "
            f"rm -f {shlex.quote(self.remote_pidfile)}"
        )
        try:
            self.remote.exec(term_cmd, timeout=15.0)
        except Exception:
            pass
        return False


def run_foreground_adb(
    remote: AdbRemote,
    cmd: Sequence[str],
    local_stdout_path: Path,
    local_stderr_path: Path,
    adb_stderr_path: Path,
    remote_log_stdout: str,
    remote_log_stderr: str,
    env_vars: Optional[Dict[str, str]] = None,
    timeout_s: float = 600.0,
) -> tuple[int, bool]:
    """Run `cmd` foreground on the device; capture remote stdout/stderr, pull
    them back to local paths.

    The remote shell wraps the command in `sh -c` with stdout/stderr
    redirected to device-side files, then we tail the rc via a sentinel
    marker file. `adb shell` historically swallows the exit code of the
    inner command on older platforms — we get it via `; echo EXIT=$?` in
    the stdout log and parse it.
    """
    env_prefix = AdbRemote._inline_env(env_vars)
    cmd_str = " ".join(shlex.quote(a) for a in cmd)
    remote_log_stdout_q = shlex.quote(remote_log_stdout)
    remote_log_stderr_q = shlex.quote(remote_log_stderr)
    rc_file = remote_log_stdout + ".rc"
    rc_file_q = shlex.quote(rc_file)

    remote_script = (
        f"{env_prefix}{cmd_str} "
        f"> {remote_log_stdout_q} 2> {remote_log_stderr_q}; "
        f"echo $? > {rc_file_q}"
    )

    adb_stderr_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    rc = -1
    with open(adb_stderr_path, "ab") as adb_err_f:
        full = remote._adb_base() + ["shell", f"sh -c {shlex.quote(remote_script)}"]
        try:
            r = subprocess.run(
                full,
                stdout=subprocess.DEVNULL,
                stderr=adb_err_f,
                timeout=timeout_s,
            )
            # adb shell rc only reflects the shell itself; read the sentinel.
            rc_rc, rc_out, _ = remote.exec(f"cat {rc_file_q} 2>/dev/null || true", timeout=10.0)
            if rc_rc == 0 and rc_out.strip().isdigit():
                rc = int(rc_out.strip())
            else:
                rc = r.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -1

    # Pull logs regardless of rc.
    try:
        remote.pull(remote_log_stdout, local_stdout_path, retries=2, timeout=120.0)
    except Exception as e:
        try:
            with open(local_stderr_path, "ab") as f:
                f.write(f"\n[run_foreground_adb] stdout pull failed: {e!r}\n".encode())
        except Exception:
            pass
    try:
        remote.pull(remote_log_stderr, local_stderr_path, retries=2, timeout=120.0)
    except Exception as e:
        try:
            with open(local_stderr_path, "ab") as f:
                f.write(f"\n[run_foreground_adb] stderr pull failed: {e!r}\n".encode())
        except Exception:
            pass

    return rc, timed_out
