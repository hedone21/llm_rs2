"""Connection abstraction for local and remote device execution."""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import ConnectionConfig


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Connection(ABC):
    """Abstract connection to a device (local, adb, ssh)."""

    @abstractmethod
    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        ...

    @abstractmethod
    def push(self, local_path: str | Path, remote_path: str) -> None:
        ...

    @abstractmethod
    def pull(self, remote_path: str, local_path: str | Path) -> None:
        ...

    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        ...

    @abstractmethod
    def mkdir(self, remote_path: str) -> None:
        ...

    @abstractmethod
    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        ...


class LocalConnection(Connection):
    """Connection for local host execution."""

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        import os

        env = dict(os.environ)
        if env_vars:
            env.update(env_vars)
        try:
            r = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return CommandResult(r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"Command timed out after {timeout}s")

    def push(self, local_path: str | Path, remote_path: str) -> None:
        src = Path(local_path)
        dst = Path(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def pull(self, remote_path: str, local_path: str | Path) -> None:
        self.push(remote_path, str(local_path))

    def file_exists(self, remote_path: str) -> bool:
        return Path(remote_path).exists()

    def mkdir(self, remote_path: str) -> None:
        Path(remote_path).mkdir(parents=True, exist_ok=True)

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        subprocess.run(["chmod", mode, remote_path], check=True)


class AdbConnection(Connection):
    """Connection via Android Debug Bridge."""

    def __init__(self, serial: str = ""):
        self._serial = serial

    def _adb_prefix(self) -> list[str]:
        if self._serial:
            return ["adb", "-s", self._serial]
        return ["adb"]

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        env_prefix = ""
        if env_vars:
            env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items()) + " "
        shell_cmd = f"{env_prefix}{command} 2>&1"
        cmd = self._adb_prefix() + ["shell", shell_cmd]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return CommandResult(r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"Command timed out after {timeout}s")

    def push(self, local_path: str | Path, remote_path: str) -> None:
        cmd = self._adb_prefix() + ["push", str(local_path), remote_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"adb push failed: {r.stderr}")

    def pull(self, remote_path: str, local_path: str | Path) -> None:
        cmd = self._adb_prefix() + ["pull", remote_path, str(local_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"adb pull failed: {r.stderr}")

    def file_exists(self, remote_path: str) -> bool:
        result = self.execute(f"ls {remote_path}")
        return result.ok

    def mkdir(self, remote_path: str) -> None:
        self.execute(f"mkdir -p {remote_path}")

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        self.execute(f"chmod {mode} {remote_path}")


class SshConnection(Connection):
    """SSH connection — interface reserved for future implementation."""

    def __init__(self, host: str = "", user: str = "", port: int = 22):
        raise NotImplementedError("SshConnection is not yet implemented")

    def execute(self, command, timeout=None, env_vars=None):
        raise NotImplementedError

    def push(self, local_path, remote_path):
        raise NotImplementedError

    def pull(self, remote_path, local_path):
        raise NotImplementedError

    def file_exists(self, remote_path):
        raise NotImplementedError

    def mkdir(self, remote_path):
        raise NotImplementedError

    def chmod(self, remote_path, mode="+x"):
        raise NotImplementedError


def create_connection(config: ConnectionConfig) -> Connection:
    """Factory: create a Connection from ConnectionConfig."""
    if config.type == "local":
        return LocalConnection()
    elif config.type == "adb":
        return AdbConnection(serial=config.serial)
    elif config.type == "ssh":
        return SshConnection()
    else:
        raise ValueError(f"Unknown connection type: {config.type}")
