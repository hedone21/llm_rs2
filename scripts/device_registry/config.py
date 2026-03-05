"""TOML configuration loader for device registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_TOML_FILENAME = "devices.toml"


@dataclass
class ConnectionConfig:
    type: str  # "local", "adb", "ssh"
    serial: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConnectionConfig:
        return cls(type=d["type"], serial=d.get("serial", ""))


@dataclass
class BuildConfig:
    target: str = ""  # empty = native build
    env_file: str = ""
    binary_dir: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BuildConfig:
        return cls(
            target=d.get("target", ""),
            env_file=d.get("env_file", ""),
            binary_dir=d.get("binary_dir", ""),
        )


@dataclass
class DevicePaths:
    work_dir: str = ""
    model_dir: str = ""
    eval_dir: str = ""
    lib_dir: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DevicePaths:
        return cls(
            work_dir=d.get("work_dir", ""),
            model_dir=d.get("model_dir", ""),
            eval_dir=d.get("eval_dir", ""),
            lib_dir=d.get("lib_dir", ""),
        )


@dataclass
class ManagerConfig:
    transport: str = ""
    thermal_zone_types: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManagerConfig:
        return cls(
            transport=d.get("transport", ""),
            thermal_zone_types=d.get("thermal_zone_types", []),
        )


@dataclass
class DeviceConfig:
    id: str
    name: str
    connection: ConnectionConfig
    build: BuildConfig
    paths: DevicePaths
    capabilities: dict[str, Any] = field(default_factory=dict)
    manager: ManagerConfig | None = None

    @classmethod
    def from_dict(cls, device_id: str, d: dict[str, Any]) -> DeviceConfig:
        conn = ConnectionConfig.from_dict(d.get("connection", {"type": "local"}))
        build = BuildConfig.from_dict(d.get("build", {}))
        paths = DevicePaths.from_dict(d.get("paths", {}))
        caps = dict(d.get("capabilities", {}))
        mgr = ManagerConfig.from_dict(d["manager"]) if "manager" in d else None
        return cls(
            id=device_id,
            name=d.get("name", device_id),
            connection=conn,
            build=build,
            paths=paths,
            capabilities=caps,
            manager=mgr,
        )

    @property
    def is_local(self) -> bool:
        return self.connection.type == "local"

    @property
    def is_android(self) -> bool:
        return self.connection.type == "adb"

    def binary_remote_path(self, binary_name: str) -> str:
        return f"{self.paths.work_dir}/{binary_name}" if self.paths.work_dir else binary_name

    def ld_library_path_env(self) -> str:
        if self.paths.lib_dir:
            return f"LD_LIBRARY_PATH={self.paths.lib_dir}"
        return ""


def find_devices_toml(start_dir: str | Path | None = None) -> Path:
    """Walk up from start_dir to find devices.toml."""
    d = Path(start_dir) if start_dir else Path.cwd()
    while True:
        candidate = d / _TOML_FILENAME
        if candidate.is_file():
            return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        f"{_TOML_FILENAME} not found in {start_dir or Path.cwd()} or any parent directory"
    )


def _load_toml(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path:
        p = Path(config_path)
    else:
        p = find_devices_toml()
    with open(p, "rb") as f:
        return tomllib.load(f)


def load_device_config(
    device_id: str, config_path: str | Path | None = None
) -> DeviceConfig:
    data = _load_toml(config_path)
    devices = data.get("devices", {})
    if device_id not in devices:
        available = ", ".join(sorted(devices.keys())) or "(none)"
        raise KeyError(f"Device '{device_id}' not found. Available: {available}")
    return DeviceConfig.from_dict(device_id, devices[device_id])


def load_all_devices(config_path: str | Path | None = None) -> dict[str, DeviceConfig]:
    data = _load_toml(config_path)
    devices = data.get("devices", {})
    return {
        dev_id: DeviceConfig.from_dict(dev_id, dev_data)
        for dev_id, dev_data in devices.items()
    }
