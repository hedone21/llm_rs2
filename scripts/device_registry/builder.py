"""Build orchestration for device targets."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import DeviceConfig


def build_binary(
    device_config: DeviceConfig,
    binary_name: str,
    project_root: str | Path,
    extra_bins: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    """Build a binary for the given device target.

    Returns True on success, False on failure.
    """
    project_root = Path(project_root)

    # Prepare environment
    env = dict(os.environ)
    if device_config.build.env_file:
        env_file = project_root / device_config.build.env_file
        if env_file.exists():
            env = _source_env_file(env_file, env)
        else:
            print(f"Warning: env_file '{env_file}' not found, skipping")

    # Build cargo command
    cmd_parts = ["cargo", "build", "--release"]

    if device_config.build.target:
        cmd_parts.extend(["--target", device_config.build.target])

    cmd_parts.extend(["--bin", binary_name])

    if extra_bins:
        for b in extra_bins:
            cmd_parts.extend(["--bin", b])

    cmd = " ".join(cmd_parts)
    print(f"  Build: {cmd}")

    if dry_run:
        print("  (dry-run: skipping build)")
        return True

    r = subprocess.run(
        cmd, shell=True, cwd=str(project_root), env=env
    )
    return r.returncode == 0


def get_local_binary_path(
    device_config: DeviceConfig,
    binary_name: str,
    project_root: str | Path,
) -> Path:
    """Get the local path to a built binary."""
    project_root = Path(project_root)
    if device_config.build.binary_dir:
        return project_root / device_config.build.binary_dir / binary_name
    # Fallback: construct from target
    if device_config.build.target:
        return project_root / "target" / device_config.build.target / "release" / binary_name
    return project_root / "target" / "release" / binary_name


def _source_env_file(env_file: Path, base_env: dict) -> dict:
    """Source a shell env file and capture the resulting environment."""
    cmd = f"set -a && source {env_file} && env"
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            env=base_env, executable="/bin/bash"
        )
        if r.returncode == 0:
            new_env = dict(base_env)
            for line in r.stdout.splitlines():
                if "=" in line:
                    key, _, val = line.partition("=")
                    new_env[key] = val
            return new_env
    except Exception:
        pass
    return base_env
