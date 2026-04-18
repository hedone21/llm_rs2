#!/usr/bin/env python3
"""Unit tests for hosts.toml config loader and toolchain env composition."""

from __future__ import annotations

import platform
import textwrap
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ package is importable when run directly
import sys
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from device_registry.config import (
    BuildConfig,
    ConnectionConfig,
    DeviceConfig,
    DevicePaths,
    HostsConfig,
    detect_current_host,
    load_hosts_config,
)
from device_registry.builder import (
    ToolchainNotFoundError,
    _compose_toolchain_env,
    _warned_env_file,
    build_binary,
)
from device_registry.connection import SshConnection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINI_HOSTS_TOML = textwrap.dedent("""\
    schema_version = 1
    default_host = "arch-linux"

    [hosts.mac-arm]
    uname_match = { sysname = "Darwin", machine = "arm64" }
      [hosts.mac-arm.toolchains.android-ndk]
      ndk_home  = "/fake/ndk-mac"
      host_tag  = "darwin-x86_64"
      api_level = 21

    [hosts.arch-linux]
    uname_match = { sysname = "Linux", machine = "x86_64" }
      [hosts.arch-linux.toolchains.android-ndk]
      ndk_home  = "/fake/ndk-linux"
      host_tag  = "linux-x86_64"
      api_level = 21

    [hosts.ubuntu-ci]
    uname_match = { sysname = "Linux", machine = "x86_64" }
    env_marker  = "CI"
      [hosts.ubuntu-ci.toolchains.android-ndk]
      ndk_home  = "${ANDROID_NDK_HOME}"
      host_tag  = "linux-x86_64"
      api_level = 21
""")


@pytest.fixture()
def hosts_toml(tmp_path: Path) -> Path:
    p = tmp_path / "hosts.toml"
    p.write_text(_MINI_HOSTS_TOML)
    return p


@pytest.fixture()
def hosts_cfg(hosts_toml: Path) -> HostsConfig:
    return load_hosts_config(hosts_toml)


# ---------------------------------------------------------------------------
# load_hosts_config
# ---------------------------------------------------------------------------

class TestLoadHostsConfig:
    def test_loads_three_hosts(self, hosts_cfg: HostsConfig) -> None:
        assert set(hosts_cfg.hosts.keys()) == {"mac-arm", "arch-linux", "ubuntu-ci"}

    def test_schema_version(self, hosts_cfg: HostsConfig) -> None:
        assert hosts_cfg.schema_version == 1

    def test_default_host(self, hosts_cfg: HostsConfig) -> None:
        assert hosts_cfg.default_host == "arch-linux"

    def test_toolchain_fields(self, hosts_cfg: HostsConfig) -> None:
        tc = hosts_cfg.hosts["mac-arm"].toolchains["android-ndk"]
        assert tc.ndk_home == "/fake/ndk-mac"
        assert tc.host_tag == "darwin-x86_64"
        assert tc.api_level == 21

    def test_env_marker(self, hosts_cfg: HostsConfig) -> None:
        assert hosts_cfg.hosts["ubuntu-ci"].env_marker == "CI"


# ---------------------------------------------------------------------------
# detect_current_host
# ---------------------------------------------------------------------------

def _make_uname(system: str, machine: str) -> platform.uname_result:
    """Create a uname_result compatible with Python 3.9+ (including 3.14+).

    Python 3.14 removed 'processor' from the constructor while keeping it in
    _fields (as a computed property). We use __new__ with the 5 positional
    constructor args: system, node, release, version, machine.
    """
    return platform.uname_result.__new__(
        platform.uname_result,
        system,   # system
        "myhost", # node
        "1.0",    # release
        "",       # version
        machine,  # machine
    )


_UnameDarwinArm = _make_uname("Darwin", "arm64")
_UnameLinuxX86 = _make_uname("Linux", "x86_64")


class TestDetectCurrentHost:
    def test_darwin_arm64(self, hosts_cfg: HostsConfig) -> None:
        with patch("platform.uname", return_value=_UnameDarwinArm):
            host = detect_current_host(hosts_cfg, env={})
        assert host.id == "mac-arm"

    def test_linux_x86_64(self, hosts_cfg: HostsConfig) -> None:
        with patch("platform.uname", return_value=_UnameLinuxX86):
            host = detect_current_host(hosts_cfg, env={})
        assert host.id == "arch-linux"

    def test_linux_x86_64_with_ci(self, hosts_cfg: HostsConfig) -> None:
        with patch("platform.uname", return_value=_UnameLinuxX86):
            host = detect_current_host(hosts_cfg, env={"CI": "true"})
        assert host.id == "ubuntu-ci"

    def test_llm_rs2_host_override(self, hosts_cfg: HostsConfig) -> None:
        # uname would normally select mac-arm, but env override wins
        with patch("platform.uname", return_value=_UnameDarwinArm):
            host = detect_current_host(hosts_cfg, env={"LLM_RS2_HOST": "arch-linux"})
        assert host.id == "arch-linux"

    def test_override_missing_key_raises(self, hosts_cfg: HostsConfig) -> None:
        with pytest.raises(RuntimeError, match="not found in hosts.toml"):
            detect_current_host(hosts_cfg, env={"LLM_RS2_HOST": "nonexistent"})

    def test_no_match_falls_back_to_default(self, hosts_cfg: HostsConfig) -> None:
        unknown_uname = _make_uname("FreeBSD", "amd64")
        with patch("platform.uname", return_value=unknown_uname):
            host = detect_current_host(hosts_cfg, env={})
        # default_host = "arch-linux"
        assert host.id == "arch-linux"

    def test_no_match_no_default_raises(self, tmp_path: Path) -> None:
        toml_content = textwrap.dedent("""\
            schema_version = 1
            [hosts.mac-arm]
            uname_match = { sysname = "Darwin", machine = "arm64" }
              [hosts.mac-arm.toolchains.android-ndk]
              ndk_home = "/fake/ndk"
              host_tag = "darwin-x86_64"
              api_level = 21
        """)
        p = tmp_path / "hosts.toml"
        p.write_text(toml_content)
        cfg = load_hosts_config(p)
        unknown_uname = _make_uname("FreeBSD", "amd64")
        with patch("platform.uname", return_value=unknown_uname):
            with pytest.raises(RuntimeError, match="set LLM_RS2_HOST="):
                detect_current_host(cfg, env={})


# ---------------------------------------------------------------------------
# _compose_toolchain_env
# ---------------------------------------------------------------------------

class TestComposeToolchainEnv:
    def test_ndk_path_in_cc(self, hosts_cfg: HostsConfig, tmp_path: Path) -> None:
        """CC_aarch64_linux_android should point to the clang binary."""
        host = hosts_cfg.hosts["mac-arm"]
        # Create a fake clang binary so existence check passes
        toolchain_bin = (
            tmp_path / "toolchains" / "llvm" / "prebuilt" / "darwin-x86_64" / "bin"
        )
        toolchain_bin.mkdir(parents=True)
        clang = toolchain_bin / "aarch64-linux-android21-clang"
        clang.touch()

        # Patch ndk_home to point to tmp_path
        host.toolchains["android-ndk"].ndk_home = str(tmp_path)

        env = _compose_toolchain_env(host, "android-ndk", "aarch64-linux-android")
        assert env["CC_aarch64_linux_android"] == str(clang)
        assert env["CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER"] == str(clang)

    def test_missing_clang_raises(self, hosts_cfg: HostsConfig, tmp_path: Path) -> None:
        """ToolchainNotFoundError if clang binary is absent."""
        host = hosts_cfg.hosts["mac-arm"]
        host.toolchains["android-ndk"].ndk_home = str(tmp_path)

        with pytest.raises(ToolchainNotFoundError, match="expected:"):
            _compose_toolchain_env(host, "android-ndk", "aarch64-linux-android")

    def test_unknown_toolchain_raises_key_error(self, hosts_cfg: HostsConfig) -> None:
        host = hosts_cfg.hosts["mac-arm"]
        with pytest.raises(KeyError, match="nonexistent"):
            _compose_toolchain_env(host, "nonexistent", "aarch64-linux-android")

    def test_env_var_expansion(self, tmp_path: Path) -> None:
        """${ANDROID_NDK_HOME} in ndk_home should be expanded."""
        toml_content = textwrap.dedent(f"""\
            schema_version = 1
            [hosts.ubuntu-ci]
            uname_match = {{ sysname = "Linux", machine = "x86_64" }}
              [hosts.ubuntu-ci.toolchains.android-ndk]
              ndk_home  = "${{ANDROID_NDK_HOME}}"
              host_tag  = "linux-x86_64"
              api_level = 21
        """)
        p = tmp_path / "hosts.toml"
        p.write_text(toml_content)

        fake_ndk = tmp_path / "fake-ndk"
        toolchain_bin = (
            fake_ndk / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "bin"
        )
        toolchain_bin.mkdir(parents=True)
        clang = toolchain_bin / "aarch64-linux-android21-clang"
        clang.touch()

        import os
        old = os.environ.get("ANDROID_NDK_HOME")
        try:
            os.environ["ANDROID_NDK_HOME"] = str(fake_ndk)
            cfg = load_hosts_config(p)
            host = cfg.hosts["ubuntu-ci"]
            assert host.toolchains["android-ndk"].ndk_home == str(fake_ndk)
        finally:
            if old is None:
                os.environ.pop("ANDROID_NDK_HOME", None)
            else:
                os.environ["ANDROID_NDK_HOME"] = old


# ---------------------------------------------------------------------------
# SSH: ConnectionConfig 파싱 테스트
# ---------------------------------------------------------------------------

class TestConnectionConfigSsh:
    def test_from_dict_basic_ssh(self) -> None:
        d = {
            "type": "ssh",
            "host": "165.132.107.73",
            "user": "nvidia",
            "port": 4121,
        }
        cfg = ConnectionConfig.from_dict(d)
        assert cfg.type == "ssh"
        assert cfg.host == "165.132.107.73"
        assert cfg.user == "nvidia"
        assert cfg.port == 4121
        assert cfg.identity_file == ""
        assert cfg.ssh_options == []

    def test_from_dict_ssh_with_identity_and_options(self) -> None:
        d = {
            "type": "ssh",
            "host": "example.com",
            "user": "admin",
            "port": 22,
            "identity_file": "~/.ssh/id_rsa",
            "ssh_options": ["-o", "StrictHostKeyChecking=no"],
        }
        cfg = ConnectionConfig.from_dict(d)
        assert cfg.identity_file == "~/.ssh/id_rsa"
        assert cfg.ssh_options == ["-o", "StrictHostKeyChecking=no"]

    def test_from_dict_ssh_default_port(self) -> None:
        cfg = ConnectionConfig.from_dict({"type": "ssh", "host": "h", "user": "u"})
        assert cfg.port == 22

    def test_adb_connection_unchanged(self) -> None:
        """기존 adb 파싱이 깨지지 않아야 함."""
        cfg = ConnectionConfig.from_dict({"type": "adb", "serial": "ABC123"})
        assert cfg.type == "adb"
        assert cfg.serial == "ABC123"
        assert cfg.host == ""


# ---------------------------------------------------------------------------
# BuildConfig: features / default_features 파싱 테스트
# ---------------------------------------------------------------------------

class TestBuildConfigFeatures:
    def test_from_dict_features_list(self) -> None:
        d = {"target": "aarch64-unknown-linux-gnu", "features": ["cuda"]}
        cfg = BuildConfig.from_dict(d)
        assert cfg.features == ["cuda"]
        assert cfg.default_features is True

    def test_from_dict_no_default_features(self) -> None:
        d = {"features": ["cuda"], "default_features": False}
        cfg = BuildConfig.from_dict(d)
        assert cfg.default_features is False

    def test_from_dict_empty_features_default(self) -> None:
        """features 미지정 시 빈 리스트, default_features=True."""
        cfg = BuildConfig.from_dict({})
        assert cfg.features == []
        assert cfg.default_features is True

    def test_existing_fields_still_work(self) -> None:
        """기존 필드(target, toolchain 등)가 그대로 동작해야 함."""
        d = {
            "target": "aarch64-linux-android",
            "toolchain": "android-ndk",
            "binary_dir": "target/aarch64-linux-android/release",
        }
        cfg = BuildConfig.from_dict(d)
        assert cfg.target == "aarch64-linux-android"
        assert cfg.toolchain == "android-ndk"
        assert cfg.features == []
        assert cfg.default_features is True


# ---------------------------------------------------------------------------
# SshConnection: _ssh_base / _scp_base 단위 테스트
# ---------------------------------------------------------------------------

class TestSshConnectionBase:
    def test_ssh_base_basic(self) -> None:
        conn = SshConnection(host="165.132.107.73", user="nvidia", port=4121)
        base = conn._ssh_base()
        assert base[0] == "ssh"
        assert "-p" in base
        assert str(4121) in base
        assert "nvidia@165.132.107.73" in base

    def test_ssh_base_with_identity_file(self) -> None:
        conn = SshConnection(
            host="165.132.107.73",
            user="nvidia",
            port=4121,
            identity_file="/home/user/.ssh/id_rsa",
        )
        base = conn._ssh_base()
        assert "-i" in base
        assert "/home/user/.ssh/id_rsa" in base

    def test_ssh_base_with_extra_options(self) -> None:
        conn = SshConnection(
            host="host.example",
            user="admin",
            port=22,
            extra_options=["-o", "StrictHostKeyChecking=no"],
        )
        base = conn._ssh_base()
        assert "-o" in base
        assert "StrictHostKeyChecking=no" in base
        # user@host은 마지막 원소여야 함
        assert base[-1] == "admin@host.example"

    def test_scp_base_uses_uppercase_P(self) -> None:
        conn = SshConnection(host="165.132.107.73", user="nvidia", port=4121)
        base = conn._scp_base()
        assert base[0] == "scp"
        assert "-P" in base
        assert str(4121) in base

    def test_scp_base_with_identity_file(self) -> None:
        conn = SshConnection(
            host="h", user="u", port=2222, identity_file="/key"
        )
        base = conn._scp_base()
        assert "-i" in base
        assert "/key" in base

    def test_ssh_connection_missing_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host"):
            SshConnection(host="", user="nvidia")

    def test_ssh_connection_missing_user_raises(self) -> None:
        with pytest.raises(ValueError, match="user"):
            SshConnection(host="165.132.107.73", user="")


# ---------------------------------------------------------------------------
# builder: features / default_features → cargo 명령 조립 검증
# ---------------------------------------------------------------------------

class TestBuilderCargoFeatureFlags:
    """build_binary dry_run 출력에서 cargo 명령 형태를 검증한다."""

    def _make_device(
        self,
        features: list[str],
        default_features: bool,
        tmp_path: Path,
    ) -> DeviceConfig:
        return DeviceConfig(
            id="test",
            name="Test Device",
            connection=ConnectionConfig(type="local"),
            build=BuildConfig(
                target="aarch64-unknown-linux-gnu",
                features=features,
                default_features=default_features,
            ),
            paths=DevicePaths(work_dir=str(tmp_path)),
        )

    def test_features_and_no_default_features_in_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        device = self._make_device(["cuda"], False, tmp_path)
        result = build_binary(device, "generate", tmp_path, dry_run=True)
        assert result is True
        captured = capsys.readouterr()
        assert "--features cuda" in captured.out
        assert "--no-default-features" in captured.out

    def test_no_flags_when_empty_features_and_default_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        device = self._make_device([], True, tmp_path)
        result = build_binary(device, "generate", tmp_path, dry_run=True)
        assert result is True
        captured = capsys.readouterr()
        assert "--features" not in captured.out
        assert "--no-default-features" not in captured.out

    def test_multiple_features_joined_by_comma(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        device = self._make_device(["cuda", "opencl"], True, tmp_path)
        build_binary(device, "generate", tmp_path, dry_run=True)
        captured = capsys.readouterr()
        assert "--features cuda,opencl" in captured.out

    def test_features_appear_before_target(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        device = self._make_device(["cuda"], False, tmp_path)
        build_binary(device, "generate", tmp_path, dry_run=True)
        captured = capsys.readouterr()
        # cargo 명령 줄 추출
        build_line = next(
            line for line in captured.out.splitlines() if "cargo build" in line
        )
        features_idx = build_line.index("--features")
        target_idx = build_line.index("--target")
        assert features_idx < target_idx


# ---------------------------------------------------------------------------
# Legacy back-compat: env_file path emits DeprecationWarning once
# ---------------------------------------------------------------------------

class TestLegacyEnvFile:
    def test_env_file_triggers_deprecation_warning(self, tmp_path: Path) -> None:
        """BuildConfig.env_file + no toolchain → DeprecationWarning."""
        import device_registry.builder as builder_mod

        # Reset module-level flag before test
        builder_mod._warned_env_file = False

        from device_registry.config import BuildConfig, DeviceConfig, ConnectionConfig, DevicePaths
        from device_registry.builder import build_binary

        # Create a minimal env file
        env_file = tmp_path / "test.source"
        env_file.write_text("export TESTVAR=1\n")

        device = DeviceConfig(
            id="test",
            name="Test",
            connection=ConnectionConfig(type="local"),
            build=BuildConfig(env_file=str(env_file.name)),
            paths=DevicePaths(work_dir=str(tmp_path)),
        )

        with pytest.warns(DeprecationWarning, match="env_file is deprecated"):
            # We use dry_run=True so no actual cargo invocation
            build_binary(device, "generate", tmp_path, dry_run=True)

        # Second call must NOT emit the warning again (module-level flag)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            build_binary(device, "generate", tmp_path, dry_run=True)

        # Cleanup flag
        builder_mod._warned_env_file = False
