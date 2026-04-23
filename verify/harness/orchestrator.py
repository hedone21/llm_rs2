"""Scenario orchestrator: baseline run + action run + 3-layer verdict.

Phase 3 additions:
- mock_manager scenario mode via `action.mock_manager_commands` (list of steps).
- YAML `delay_sec` is converted to JSON `delay_ms` (×1000) for mock_manager.
- `baseline.extra_args` flows verbatim to both baseline and action runs.
- Heartbeats (full / short / compact) are parsed and fed to assertions.

Phase 4 additions:
- SSH remote orchestration (Jetson) via `connection.type == "ssh"`.
- TCP transport (127.0.0.1:9100) enforced for SSH path — Unix sockets would
  collide with shared user accounts and bind-permission issues.
- Remote work-dir scheme: `<work_dir>/verify_runs/<scenario>_r<idx>/`.
- Prompt + scenario JSON pushed to remote via scp.
- Baseline & action stderr files pulled back to local for the existing
  assertion code to parse.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .assertions import (
    aggregate_verdict,
    verify_accuracy,
    verify_crash_and_progress,
    verify_functional,
    verify_performance,
)
from .fixtures import (
    PROJECT_ROOT,
    load_models_toml,
    load_prompt,
    resolve_model_entry,
)
from .log_parser import load_summary, parse_heartbeats
from .process_control import (
    AdbBackgroundProcess,
    AdbRemote,
    BackgroundProcess,
    SshBackgroundProcess,
    adb_remote_from_dict,
    pkill_pattern,
    remove_socket_files,
    resolve_adb_serial,
    run_foreground,
    run_foreground_adb,
    run_foreground_ssh,
    ssh_remote_from_dict,
)
from .spec_loader import ScenarioSpec
from .text_accuracy import decode_jsonl_to_text


# ── Binary paths ────────────────────────────────────────

def _host_binary(device_cfg: Dict[str, Any], name: str) -> Path:
    binary_dir = device_cfg.get("build", {}).get("binary_dir", "target/release")
    path = PROJECT_ROOT / binary_dir / name
    return path


def _remote_binary_path(device_cfg: Dict[str, Any], name: str) -> str:
    work_dir = device_cfg.get("paths", {}).get("work_dir", "")
    if not work_dir:
        raise KeyError("device paths.work_dir is required for remote execution")
    return f"{work_dir}/{name}"


def _resolve_backend(cfg_backend: str, device_cfg: Dict[str, Any]) -> str:
    if cfg_backend != "auto":
        return cfg_backend
    gpu = device_cfg.get("capabilities", {}).get("gpu", "none")
    if gpu == "none":
        return "cpu"
    if "cuda" in str(gpu).lower() or "nvidia" in str(gpu).lower():
        return "cuda"
    return "opencl"


# ── Command builders ────────────────────────────────────

def _generate_cmd(
    generate_bin: str,
    model_path: str,
    prompt_file: str,
    experiment_output: str,
    decode_tokens: int,
    prefill_chunk_size: int,
    backend: str,
    enable_resilience: bool,
    resilience_transport: Optional[str],
    extra_args: list,
) -> list:
    cmd = [
        str(generate_bin),
        "--model-path", model_path,
        "--prompt-file", str(prompt_file),
        "--num-tokens", str(decode_tokens),
        "--greedy",
        "--ignore-eos",
        "--experiment-output", str(experiment_output),
        "--prefill-chunk-size", str(prefill_chunk_size),
        "--backend", backend,
    ]
    if enable_resilience:
        cmd.append("--enable-resilience")
        if resilience_transport:
            cmd.extend(["--resilience-transport", resilience_transport])
    cmd.extend(extra_args)
    return cmd


def _mock_manager_single_cmd_local(
    mock_bin: Path,
    socket_path: Path,
    command: str,
    params: Dict[str, Any],
    wait_secs: int,
) -> list:
    cmd = [
        str(mock_bin),
        "--socket", str(socket_path),
        "--command", command,
        "--wait-secs", str(wait_secs),
    ]
    if "delay_ms" in params:
        cmd.extend(["--delay-ms", str(params["delay_ms"])])
    if "keep_ratio" in params:
        cmd.extend(["--keep-ratio", str(params["keep_ratio"])])
    if "sink_size" in params:
        cmd.extend(["--sink-size", str(params["sink_size"])])
    if "window_size" in params:
        cmd.extend(["--window-size", str(params["window_size"])])
    if "device" in params:
        cmd.extend(["--device", str(params["device"])])
    if "target_bits" in params:
        cmd.extend(["--target-bits", str(params["target_bits"])])
    if "target_ms" in params:
        cmd.extend(["--target-ms", str(params["target_ms"])])
    if "ratio" in params:
        cmd.extend(["--ratio", str(params["ratio"])])
    return cmd


def _mock_manager_single_cmd_tcp(
    mock_bin: str,
    tcp_addr: str,
    command: str,
    params: Dict[str, Any],
    wait_secs: int,
) -> list:
    cmd = [
        str(mock_bin),
        "--tcp", tcp_addr,
        "--command", command,
        "--wait-secs", str(wait_secs),
    ]
    if "delay_ms" in params:
        cmd.extend(["--delay-ms", str(params["delay_ms"])])
    if "keep_ratio" in params:
        cmd.extend(["--keep-ratio", str(params["keep_ratio"])])
    if "sink_size" in params:
        cmd.extend(["--sink-size", str(params["sink_size"])])
    if "window_size" in params:
        cmd.extend(["--window-size", str(params["window_size"])])
    if "device" in params:
        cmd.extend(["--device", str(params["device"])])
    if "target_bits" in params:
        cmd.extend(["--target-bits", str(params["target_bits"])])
    if "target_ms" in params:
        cmd.extend(["--target-ms", str(params["target_ms"])])
    if "ratio" in params:
        cmd.extend(["--ratio", str(params["ratio"])])
    return cmd


def _mock_manager_scenario_cmd_local(
    mock_bin: Path,
    socket_path: Path,
    scenario_file: Path,
    wait_secs: int,
) -> list:
    return [
        str(mock_bin),
        "--socket", str(socket_path),
        "--scenario", str(scenario_file),
        "--wait-secs", str(wait_secs),
    ]


def _mock_manager_scenario_cmd_tcp(
    mock_bin: str,
    tcp_addr: str,
    scenario_file: str,
    wait_secs: int,
) -> list:
    return [
        str(mock_bin),
        "--tcp", tcp_addr,
        "--scenario", str(scenario_file),
        "--wait-secs", str(wait_secs),
    ]


# ── Scenario JSON building ──────────────────────────────

_SCENARIO_PARAM_KEYS = {
    "delay_ms", "delay_ms_param",
    "keep_ratio", "sink_size", "window_size",
    "device", "target_bits", "skip_ratio", "ratio",
    "chunk_size", "yield_ms", "cpu_chunk_size",
}


def _build_scenario_json(spec: ScenarioSpec) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    for entry in spec.action.mock_manager_commands or []:
        delay_sec = float(entry.get("delay_sec", 0.0))
        delay_ms = int(round(delay_sec * 1000.0))
        command = str(entry["command"])
        params = dict(entry.get("params", {}) or {})

        step: Dict[str, Any] = {"delay_ms": delay_ms, "command": command}

        if command == "Throttle" and "delay_ms" in params:
            step["delay_ms_param"] = int(params["delay_ms"])
        if command == "SetTargetTbt" and "target_ms" in params:
            step["delay_ms_param"] = int(params["target_ms"])

        for k, v in params.items():
            if k in _SCENARIO_PARAM_KEYS and k != "delay_ms":
                step[k] = v
            elif k == "delay_ms" and command != "Throttle":
                step["delay_ms_param"] = int(v)

        steps.append(step)

    return {
        "name": spec.id,
        "description": spec.description,
        "commands": steps,
    }


def _scenario_total_delay_sec(spec: ScenarioSpec) -> float:
    total = 0.0
    for entry in spec.action.mock_manager_commands or []:
        total += float(entry.get("delay_sec", 0.0))
    return total


# ── Entry points ───────────────────────────────────────


def run_scenario(
    spec: ScenarioSpec,
    device_cfg: Dict[str, Any],
    model_key: str,
    out_dir: Path,
    run_idx: int,
) -> Dict[str, Any]:
    """Run one (scenario, device, model, run) combo and return verdict dict."""
    conn_type = device_cfg.get("connection", {}).get("type", "local")

    if spec.layer == "signal":
        if conn_type == "adb":
            return _run_scenario_adb_signal(spec, device_cfg, model_key, out_dir, run_idx)
        raise NotImplementedError(
            f"layer=signal is currently only wired for adb (got connection.type={conn_type})"
        )
    if spec.layer != "engine_cmd":
        raise NotImplementedError(f"Unknown layer: {spec.layer}")

    if conn_type == "local":
        return _run_scenario_local(spec, device_cfg, model_key, out_dir, run_idx)
    if conn_type == "ssh":
        return _run_scenario_ssh(spec, device_cfg, model_key, out_dir, run_idx)
    if conn_type == "adb":
        return _run_scenario_adb(spec, device_cfg, model_key, out_dir, run_idx)
    raise NotImplementedError(f"Unsupported connection.type: {conn_type}")


# ── Local branch (Phase 1-3, unchanged) ────────────────


def _run_scenario_local(
    spec: ScenarioSpec,
    device_cfg: Dict[str, Any],
    model_key: str,
    out_dir: Path,
    run_idx: int,
) -> Dict[str, Any]:
    device_key = device_cfg.get("_key", "host")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_toml = load_models_toml()
    model_entry = resolve_model_entry(models_toml, device_key, model_key)
    model_path = model_entry["model_path"]
    tokenizer_path = Path(model_entry["tokenizer_path"])
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)
    if not tokenizer_path.is_absolute():
        tokenizer_path = PROJECT_ROOT / tokenizer_path

    generate_bin = _host_binary(device_cfg, "generate")
    mock_bin = _host_binary(device_cfg, "mock_manager")
    if not generate_bin.exists():
        raise FileNotFoundError(f"generate binary missing: {generate_bin}")
    if not mock_bin.exists():
        raise FileNotFoundError(f"mock_manager binary missing: {mock_bin}")

    backend = _resolve_backend(spec.baseline.backend, device_cfg)

    prompt_text = load_prompt(spec.baseline.prompt)
    prompt_file = out_dir / "prompt.txt"
    prompt_file.write_text(prompt_text, encoding="utf-8")

    pkill_pattern("mock_manager --socket /tmp/verify_")
    remove_socket_files(["/tmp/verify_*.sock"])

    baseline_jsonl = out_dir / "baseline.jsonl"
    baseline_stderr = out_dir / "baseline.stderr"
    baseline_stdout = out_dir / "baseline.stdout"
    action_jsonl = out_dir / "action.jsonl"
    action_stderr = out_dir / "action.stderr"
    action_stdout = out_dir / "action.stdout"
    manager_stdout = out_dir / "manager.stdout"
    manager_stderr = out_dir / "manager.stderr"
    verdict_path = out_dir / "verdict.json"

    timings: Dict[str, float] = {}

    # ── Step 1: baseline ────────────────────────────────
    baseline_cmd = _generate_cmd(
        generate_bin=str(generate_bin),
        model_path=model_path,
        prompt_file=str(prompt_file),
        experiment_output=str(baseline_jsonl),
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=False,
        resilience_transport=None,
        extra_args=list(spec.baseline.extra_args),
    )
    t0 = time.monotonic()
    rc_b, timed_b = run_foreground(
        baseline_cmd, baseline_stdout, baseline_stderr, timeout_s=600.0
    )
    baseline_wall = time.monotonic() - t0
    timings["baseline_wall_s"] = baseline_wall

    # ── Step 2: action ──────────────────────────────────
    socket_path = Path(f"/tmp/verify_{os.getpid()}_{spec.id}.sock")
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    use_scenario_mode = bool(spec.action.mock_manager_commands)
    use_single_mode = spec.action.mock_manager_command is not None

    if not use_scenario_mode and not use_single_mode:
        raise ValueError(
            f"Scenario {spec.id} has neither mock_manager_commands nor "
            "mock_manager_command in action"
        )

    scenario_file: Optional[Path] = None
    if use_scenario_mode:
        scenario_file = out_dir / "scenario.json"
        scenario_payload = _build_scenario_json(spec)
        scenario_file.write_text(json.dumps(scenario_payload, indent=2), encoding="utf-8")
        wait_secs = int(spec.action.mock_manager_params.get("wait_secs", 1))
        mock_cmd = _mock_manager_scenario_cmd_local(
            mock_bin=mock_bin,
            socket_path=socket_path,
            scenario_file=scenario_file,
            wait_secs=wait_secs,
        )
        total_delay = _scenario_total_delay_sec(spec)
        action_timeout = max(
            180.0,
            baseline_wall * 1.5 + wait_secs + total_delay + 60.0,
        )
    else:
        params = dict(spec.action.mock_manager_params or {})
        wait_secs = int(params.pop("wait_secs", 2))
        mock_cmd = _mock_manager_single_cmd_local(
            mock_bin=mock_bin,
            socket_path=socket_path,
            command=spec.action.mock_manager_command,
            params=params,
            wait_secs=wait_secs,
        )
        action_timeout = max(180.0, baseline_wall * 1.5 + wait_secs + 60.0)

    action_cmd = _generate_cmd(
        generate_bin=str(generate_bin),
        model_path=model_path,
        prompt_file=str(prompt_file),
        experiment_output=str(action_jsonl),
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=True,
        resilience_transport=f"unix:{socket_path}",
        extra_args=list(spec.baseline.extra_args),
    )

    t1 = time.monotonic()
    rc_a = None
    timed_a = False
    with BackgroundProcess(mock_cmd, manager_stdout, manager_stderr) as mock_proc:
        for _ in range(50):
            if socket_path.exists():
                break
            if mock_proc.poll() is not None:
                break
            time.sleep(0.1)
        time.sleep(0.5)

        rc_a, timed_a = run_foreground(
            action_cmd, action_stdout, action_stderr, timeout_s=action_timeout
        )
        mock_proc.wait(timeout=10.0)

    timings["action_wall_s"] = time.monotonic() - t1

    return _finalize_verdict(
        spec=spec,
        device_key=device_key,
        model_key=model_key,
        run_idx=run_idx,
        backend=backend,
        use_scenario_mode=use_scenario_mode,
        scenario_file=scenario_file,
        timings=timings,
        baseline_jsonl=baseline_jsonl,
        action_jsonl=action_jsonl,
        action_stderr=action_stderr,
        manager_stdout=manager_stdout,
        out_dir=out_dir,
        rc_b=rc_b, timed_b=timed_b,
        rc_a=rc_a, timed_a=timed_a,
        tokenizer_path=tokenizer_path,
        verdict_path=verdict_path,
    )


# ── SSH branch (Phase 4) ───────────────────────────────


def _run_scenario_ssh(
    spec: ScenarioSpec,
    device_cfg: Dict[str, Any],
    model_key: str,
    out_dir: Path,
    run_idx: int,
) -> Dict[str, Any]:
    device_key = device_cfg.get("_key", "jetson")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_toml = load_models_toml()
    model_entry = resolve_model_entry(models_toml, device_key, model_key)
    # `model_path` is a REMOTE path.
    remote_model_path = model_entry["model_path"]

    # tokenizer is always local
    tokenizer_path = Path(model_entry["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = PROJECT_ROOT / tokenizer_path

    # Build ssh remote helper
    remote = ssh_remote_from_dict(device_cfg.get("connection", {}))

    # Remote binaries (must already be deployed — verify_resilience.py handles it)
    generate_bin = _remote_binary_path(device_cfg, "generate")
    mock_bin = _remote_binary_path(device_cfg, "mock_manager")

    # Remote workdir per-run
    remote_work_root = device_cfg.get("paths", {}).get("work_dir", "/tmp") \
        + "/verify_runs"
    remote_run_dir = f"{remote_work_root}/{spec.id}_r{run_idx}_{os.getpid()}"
    remote.exec(f"mkdir -p {shlex.quote(remote_run_dir)}", timeout=15.0)

    # Pre-clean any stale processes / sockets on the remote host.
    # kill errors are ignored (best effort).
    remote.exec(
        "pkill -TERM -f 'llm_manager' 2>/dev/null; "
        "pkill -TERM -f 'mock_manager' 2>/dev/null; "
        "pkill -TERM -f '/home/nvidia/llm_rs2/generate' 2>/dev/null; "
        "sleep 0.3; "
        "pkill -KILL -f 'llm_manager' 2>/dev/null; "
        "pkill -KILL -f 'mock_manager' 2>/dev/null; "
        "pkill -KILL -f '/home/nvidia/llm_rs2/generate' 2>/dev/null; "
        "rm -f /tmp/verify_*.sock 2>/dev/null; "
        "true",
        timeout=20.0,
    )

    backend = _resolve_backend(spec.baseline.backend, device_cfg)

    # CUDA library path for Jetson
    lib_dir = device_cfg.get("paths", {}).get("lib_dir", "")
    env_vars: Dict[str, str] = {}
    if lib_dir:
        env_vars["LD_LIBRARY_PATH"] = f"{lib_dir}:$LD_LIBRARY_PATH"

    # Prompt file — write locally then scp to remote
    prompt_text = load_prompt(spec.baseline.prompt)
    local_prompt = out_dir / "prompt.txt"
    local_prompt.write_text(prompt_text, encoding="utf-8")
    remote_prompt = f"{remote_run_dir}/prompt.txt"
    remote.push(local_prompt, remote_prompt, retries=2, timeout=30.0)

    # Local and remote file layout
    baseline_jsonl_local = out_dir / "baseline.jsonl"
    baseline_stdout_local = out_dir / "baseline.stdout"
    baseline_stderr_local = out_dir / "baseline.stderr"
    action_jsonl_local = out_dir / "action.jsonl"
    action_stdout_local = out_dir / "action.stdout"
    action_stderr_local = out_dir / "action.stderr"
    manager_stdout_local = out_dir / "manager.stdout"
    manager_stderr_local = out_dir / "manager.stderr"
    baseline_ssh_stderr = out_dir / "baseline.ssh.stderr"
    action_ssh_stderr = out_dir / "action.ssh.stderr"
    manager_ssh_stderr = out_dir / "manager.ssh.stderr"
    verdict_path = out_dir / "verdict.json"

    baseline_jsonl_remote = f"{remote_run_dir}/baseline.jsonl"
    baseline_stdout_remote = f"{remote_run_dir}/baseline.stdout"
    baseline_stderr_remote = f"{remote_run_dir}/baseline.stderr"
    action_jsonl_remote = f"{remote_run_dir}/action.jsonl"
    action_stdout_remote = f"{remote_run_dir}/action.stdout"
    action_stderr_remote = f"{remote_run_dir}/action.stderr"
    manager_stdout_remote = f"{remote_run_dir}/manager.stdout"
    manager_pidfile_remote = f"{remote_run_dir}/mock_manager.pid"

    timings: Dict[str, float] = {}

    # ── Step 1: baseline ────────────────────────────────
    baseline_cmd = _generate_cmd(
        generate_bin=generate_bin,
        model_path=remote_model_path,
        prompt_file=remote_prompt,
        experiment_output=baseline_jsonl_remote,
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=False,
        resilience_transport=None,
        extra_args=list(spec.baseline.extra_args),
    )
    t0 = time.monotonic()
    rc_b, timed_b = run_foreground_ssh(
        remote=remote,
        cmd=baseline_cmd,
        local_stdout_path=baseline_stdout_local,
        local_stderr_path=baseline_stderr_local,
        ssh_stderr_path=baseline_ssh_stderr,
        remote_log_stdout=baseline_stdout_remote,
        remote_log_stderr=baseline_stderr_remote,
        env_vars=env_vars,
        timeout_s=600.0,
    )
    baseline_wall = time.monotonic() - t0
    timings["baseline_wall_s"] = baseline_wall

    # Pull the experiment JSONL
    try:
        remote.pull(baseline_jsonl_remote, baseline_jsonl_local, retries=2, timeout=60.0)
    except Exception as e:
        print(f"  [warn] baseline.jsonl pull failed: {e}")
        if not baseline_jsonl_local.exists():
            baseline_jsonl_local.write_bytes(b"")

    # ── Step 2: action ──────────────────────────────────
    use_scenario_mode = bool(spec.action.mock_manager_commands)
    use_single_mode = spec.action.mock_manager_command is not None
    if not use_scenario_mode and not use_single_mode:
        raise ValueError(
            f"Scenario {spec.id} has neither mock_manager_commands nor "
            "mock_manager_command"
        )

    # Jetson forces TCP transport (no Unix socket permission dance).
    tcp_addr = "127.0.0.1:9100"

    scenario_file_local: Optional[Path] = None
    scenario_file_remote: Optional[str] = None
    if use_scenario_mode:
        scenario_file_local = out_dir / "scenario.json"
        scenario_payload = _build_scenario_json(spec)
        scenario_file_local.write_text(
            json.dumps(scenario_payload, indent=2), encoding="utf-8"
        )
        scenario_file_remote = f"{remote_run_dir}/scenario.json"
        remote.push(scenario_file_local, scenario_file_remote, retries=2, timeout=30.0)

        wait_secs = int(spec.action.mock_manager_params.get("wait_secs", 1))
        mock_cmd = _mock_manager_scenario_cmd_tcp(
            mock_bin=mock_bin,
            tcp_addr=tcp_addr,
            scenario_file=scenario_file_remote,
            wait_secs=wait_secs,
        )
        total_delay = _scenario_total_delay_sec(spec)
        action_timeout = max(
            240.0,
            baseline_wall * 1.5 + wait_secs + total_delay + 90.0,
        )
    else:
        params = dict(spec.action.mock_manager_params or {})
        wait_secs = int(params.pop("wait_secs", 2))
        mock_cmd = _mock_manager_single_cmd_tcp(
            mock_bin=mock_bin,
            tcp_addr=tcp_addr,
            command=spec.action.mock_manager_command,
            params=params,
            wait_secs=wait_secs,
        )
        action_timeout = max(240.0, baseline_wall * 1.5 + wait_secs + 90.0)

    action_cmd = _generate_cmd(
        generate_bin=generate_bin,
        model_path=remote_model_path,
        prompt_file=remote_prompt,
        experiment_output=action_jsonl_remote,
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=True,
        resilience_transport=f"tcp:{tcp_addr}",
        extra_args=list(spec.baseline.extra_args),
    )

    t1 = time.monotonic()
    rc_a: Optional[int] = None
    timed_a = False
    with SshBackgroundProcess(
        remote=remote,
        cmd=mock_cmd,
        remote_log=manager_stdout_remote,
        remote_pidfile=manager_pidfile_remote,
        ssh_stderr_log=manager_ssh_stderr,
        env_vars=None,  # mock_manager doesn't need CUDA
        startup_wait_s=0.5,
    ) as mock_proc:
        # Let mock_manager bind the TCP listener before engine tries to connect.
        time.sleep(1.0)

        rc_a, timed_a = run_foreground_ssh(
            remote=remote,
            cmd=action_cmd,
            local_stdout_path=action_stdout_local,
            local_stderr_path=action_stderr_local,
            ssh_stderr_path=action_ssh_stderr,
            remote_log_stdout=action_stdout_remote,
            remote_log_stderr=action_stderr_remote,
            env_vars=env_vars,
            timeout_s=action_timeout,
        )
        # Small grace period so mock_manager flushes its observe-loop heartbeats
        # before we pull the log.
        time.sleep(1.0)

    # Pull mock_manager log (even if the process was killed via __exit__)
    try:
        remote.pull(
            manager_stdout_remote, manager_stdout_local, retries=2, timeout=30.0
        )
    except Exception as e:
        print(f"  [warn] manager.stdout pull failed: {e}")
        if not manager_stdout_local.exists():
            manager_stdout_local.write_bytes(b"")

    # Pull action JSONL
    try:
        remote.pull(action_jsonl_remote, action_jsonl_local, retries=2, timeout=60.0)
    except Exception as e:
        print(f"  [warn] action.jsonl pull failed: {e}")
        if not action_jsonl_local.exists():
            action_jsonl_local.write_bytes(b"")

    # manager.stderr is not separately captured remotely (nohup merges 2>&1).
    # Leave the local file empty for compatibility with older callers.
    if not manager_stderr_local.exists():
        manager_stderr_local.write_bytes(b"")

    timings["action_wall_s"] = time.monotonic() - t1

    return _finalize_verdict(
        spec=spec,
        device_key=device_key,
        model_key=model_key,
        run_idx=run_idx,
        backend=backend,
        use_scenario_mode=use_scenario_mode,
        scenario_file=scenario_file_local,
        timings=timings,
        baseline_jsonl=baseline_jsonl_local,
        action_jsonl=action_jsonl_local,
        action_stderr=action_stderr_local,
        manager_stdout=manager_stdout_local,
        out_dir=out_dir,
        rc_b=rc_b, timed_b=timed_b,
        rc_a=rc_a, timed_a=timed_a,
        tokenizer_path=tokenizer_path,
        verdict_path=verdict_path,
    )


# ── ADB branch (Phase 5) ───────────────────────────────


def _run_scenario_adb(
    spec: ScenarioSpec,
    device_cfg: Dict[str, Any],
    model_key: str,
    out_dir: Path,
    run_idx: int,
) -> Dict[str, Any]:
    device_key = device_cfg.get("_key", "galaxy_s25")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_toml = load_models_toml()
    model_entry = resolve_model_entry(models_toml, device_key, model_key)
    remote_model_path = model_entry["model_path"]

    tokenizer_path = Path(model_entry["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = PROJECT_ROOT / tokenizer_path

    # Resolve adb serial (honour explicit, else autodetect the single attached).
    conn_cfg = dict(device_cfg.get("connection", {}) or {})
    serial = resolve_adb_serial(conn_cfg.get("serial", "") or "")
    conn_cfg["serial"] = serial
    remote = adb_remote_from_dict(conn_cfg)

    # Remote binaries (assumed deployed by verify_resilience.py)
    generate_bin = _remote_binary_path(device_cfg, "generate")
    mock_bin = _remote_binary_path(device_cfg, "mock_manager")

    # Remote workdir per-run
    remote_work_root = device_cfg.get("paths", {}).get("work_dir", "/data/local/tmp") \
        + "/verify_runs"
    remote_run_dir = f"{remote_work_root}/{spec.id}_r{run_idx}_{os.getpid()}"
    remote.exec(f"mkdir -p {shlex.quote(remote_run_dir)}", timeout=15.0)

    # Pre-clean stale processes / sockets on device.
    work_dir = device_cfg.get("paths", {}).get("work_dir", "/data/local/tmp")
    remote.exec(
        "pkill -TERM -f 'llm_manager' 2>/dev/null; "
        "pkill -TERM -f 'mock_manager' 2>/dev/null; "
        f"pkill -TERM -f '{work_dir}/generate' 2>/dev/null; "
        "sleep 0.3; "
        "pkill -KILL -f 'llm_manager' 2>/dev/null; "
        "pkill -KILL -f 'mock_manager' 2>/dev/null; "
        f"pkill -KILL -f '{work_dir}/generate' 2>/dev/null; "
        f"rm -f {work_dir}/verify_*.sock 2>/dev/null; "
        "true",
        timeout=20.0,
    )

    backend = _resolve_backend(spec.baseline.backend, device_cfg)

    lib_dir = device_cfg.get("paths", {}).get("lib_dir", "")
    env_vars: Dict[str, str] = {
        "RAYON_NUM_THREADS": "6",  # Galaxy S25 mandated (CLAUDE.md)
    }
    if lib_dir:
        env_vars["LD_LIBRARY_PATH"] = lib_dir

    # Generate reads `--threads` (not the RAYON env var) for pool sizing.
    # Force 6T on Galaxy S25 per CLAUDE.md unless the scenario already set it.
    forced_thread_args: list = []
    scenario_extra = list(spec.baseline.extra_args)
    if "--threads" not in scenario_extra:
        forced_thread_args = ["--threads", "6"]

    # Prompt file — write locally then push
    prompt_text = load_prompt(spec.baseline.prompt)
    local_prompt = out_dir / "prompt.txt"
    local_prompt.write_text(prompt_text, encoding="utf-8")
    remote_prompt = f"{remote_run_dir}/prompt.txt"
    remote.push(local_prompt, remote_prompt, retries=2, timeout=60.0)

    baseline_jsonl_local = out_dir / "baseline.jsonl"
    baseline_stdout_local = out_dir / "baseline.stdout"
    baseline_stderr_local = out_dir / "baseline.stderr"
    action_jsonl_local = out_dir / "action.jsonl"
    action_stdout_local = out_dir / "action.stdout"
    action_stderr_local = out_dir / "action.stderr"
    manager_stdout_local = out_dir / "manager.stdout"
    manager_stderr_local = out_dir / "manager.stderr"
    baseline_adb_stderr = out_dir / "baseline.adb.stderr"
    action_adb_stderr = out_dir / "action.adb.stderr"
    manager_adb_stderr = out_dir / "manager.adb.stderr"
    verdict_path = out_dir / "verdict.json"

    baseline_jsonl_remote = f"{remote_run_dir}/baseline.jsonl"
    baseline_stdout_remote = f"{remote_run_dir}/baseline.stdout"
    baseline_stderr_remote = f"{remote_run_dir}/baseline.stderr"
    action_jsonl_remote = f"{remote_run_dir}/action.jsonl"
    action_stdout_remote = f"{remote_run_dir}/action.stdout"
    action_stderr_remote = f"{remote_run_dir}/action.stderr"
    manager_stdout_remote = f"{remote_run_dir}/manager.stdout"
    manager_pidfile_remote = f"{remote_run_dir}/mock_manager.pid"

    timings: Dict[str, float] = {}

    # ── Step 1: baseline ────────────────────────────────
    baseline_cmd = _generate_cmd(
        generate_bin=generate_bin,
        model_path=remote_model_path,
        prompt_file=remote_prompt,
        experiment_output=baseline_jsonl_remote,
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=False,
        resilience_transport=None,
        extra_args=scenario_extra + forced_thread_args,
    )
    t0 = time.monotonic()
    # S25 can be slower than Jetson; bump the baseline timeout generously.
    rc_b, timed_b = run_foreground_adb(
        remote=remote,
        cmd=baseline_cmd,
        local_stdout_path=baseline_stdout_local,
        local_stderr_path=baseline_stderr_local,
        adb_stderr_path=baseline_adb_stderr,
        remote_log_stdout=baseline_stdout_remote,
        remote_log_stderr=baseline_stderr_remote,
        env_vars=env_vars,
        timeout_s=900.0,
    )
    baseline_wall = time.monotonic() - t0
    timings["baseline_wall_s"] = baseline_wall

    try:
        remote.pull(baseline_jsonl_remote, baseline_jsonl_local, retries=2, timeout=120.0)
    except Exception as e:
        print(f"  [warn] baseline.jsonl pull failed: {e}")
        if not baseline_jsonl_local.exists():
            baseline_jsonl_local.write_bytes(b"")

    # ── Step 2: action ──────────────────────────────────
    use_scenario_mode = bool(spec.action.mock_manager_commands)
    use_single_mode = spec.action.mock_manager_command is not None
    if not use_scenario_mode and not use_single_mode:
        raise ValueError(
            f"Scenario {spec.id} has neither mock_manager_commands nor "
            "mock_manager_command"
        )

    # Android SELinux blocks filesystem unix-socket bind under /data/local/tmp,
    # so we use TCP (same as Jetson path).
    tcp_addr = "127.0.0.1:9100"
    transport_engine = f"tcp:{tcp_addr}"

    scenario_file_local: Optional[Path] = None
    scenario_file_remote: Optional[str] = None
    if use_scenario_mode:
        scenario_file_local = out_dir / "scenario.json"
        scenario_payload = _build_scenario_json(spec)
        scenario_file_local.write_text(
            json.dumps(scenario_payload, indent=2), encoding="utf-8"
        )
        scenario_file_remote = f"{remote_run_dir}/scenario.json"
        remote.push(scenario_file_local, scenario_file_remote, retries=2, timeout=60.0)

        wait_secs = int(spec.action.mock_manager_params.get("wait_secs", 1))
        mock_cmd = _mock_manager_scenario_cmd_tcp(
            mock_bin=mock_bin,
            tcp_addr=tcp_addr,
            scenario_file=scenario_file_remote,
            wait_secs=wait_secs,
        )
        total_delay = _scenario_total_delay_sec(spec)
        action_timeout = max(
            360.0,
            baseline_wall * 2.0 + wait_secs + total_delay + 120.0,
        )
    else:
        params = dict(spec.action.mock_manager_params or {})
        wait_secs = int(params.pop("wait_secs", 2))
        mock_cmd = _mock_manager_single_cmd_tcp(
            mock_bin=mock_bin,
            tcp_addr=tcp_addr,
            command=spec.action.mock_manager_command,
            params=params,
            wait_secs=wait_secs,
        )
        action_timeout = max(360.0, baseline_wall * 2.0 + wait_secs + 120.0)

    action_cmd = _generate_cmd(
        generate_bin=generate_bin,
        model_path=remote_model_path,
        prompt_file=remote_prompt,
        experiment_output=action_jsonl_remote,
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=True,
        resilience_transport=transport_engine,
        extra_args=scenario_extra + forced_thread_args,
    )

    t1 = time.monotonic()
    rc_a: Optional[int] = None
    timed_a = False

    with AdbBackgroundProcess(
        remote=remote,
        cmd=mock_cmd,
        remote_log=manager_stdout_remote,
        remote_pidfile=manager_pidfile_remote,
        adb_stderr_log=manager_adb_stderr,
        env_vars=None,  # mock_manager doesn't need LD_LIBRARY_PATH
        startup_wait_s=0.5,
    ) as mock_proc:
        # Allow mock_manager to bind TCP listener before the engine connects.
        time.sleep(1.5)

        rc_a, timed_a = run_foreground_adb(
            remote=remote,
            cmd=action_cmd,
            local_stdout_path=action_stdout_local,
            local_stderr_path=action_stderr_local,
            adb_stderr_path=action_adb_stderr,
            remote_log_stdout=action_stdout_remote,
            remote_log_stderr=action_stderr_remote,
            env_vars=env_vars,
            timeout_s=action_timeout,
        )
        time.sleep(1.0)  # grace period for mock_manager to flush heartbeats

    # Pull mock_manager log
    try:
        remote.pull(manager_stdout_remote, manager_stdout_local, retries=2, timeout=60.0)
    except Exception as e:
        print(f"  [warn] manager.stdout pull failed: {e}")
        if not manager_stdout_local.exists():
            manager_stdout_local.write_bytes(b"")

    try:
        remote.pull(action_jsonl_remote, action_jsonl_local, retries=2, timeout=120.0)
    except Exception as e:
        print(f"  [warn] action.jsonl pull failed: {e}")
        if not action_jsonl_local.exists():
            action_jsonl_local.write_bytes(b"")

    # manager.stderr merged into stdout via 2>&1 — keep empty for compat.
    if not manager_stderr_local.exists():
        manager_stderr_local.write_bytes(b"")

    timings["action_wall_s"] = time.monotonic() - t1

    return _finalize_verdict(
        spec=spec,
        device_key=device_key,
        model_key=model_key,
        run_idx=run_idx,
        backend=backend,
        use_scenario_mode=use_scenario_mode,
        scenario_file=scenario_file_local,
        timings=timings,
        baseline_jsonl=baseline_jsonl_local,
        action_jsonl=action_jsonl_local,
        action_stderr=action_stderr_local,
        manager_stdout=manager_stdout_local,
        out_dir=out_dir,
        rc_b=rc_b, timed_b=timed_b,
        rc_a=rc_a, timed_a=timed_a,
        tokenizer_path=tokenizer_path,
        verdict_path=verdict_path,
    )


# ── ADB signal branch (Phase v2-3) ─────────────────────


def _adb_forward(local_port: int, device_port: int, serial: str) -> None:
    import subprocess
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["forward", f"tcp:{local_port}", f"tcp:{device_port}"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=10.0)


def _adb_forward_remove(local_port: int, serial: str) -> None:
    import subprocess
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["forward", "--remove", f"tcp:{local_port}"]
    subprocess.run(cmd, capture_output=True, timeout=10.0)


def _wait_for_remote_log_pattern(
    remote: "AdbRemote",
    remote_path: str,
    pattern: str,
    timeout_s: float,
    poll_s: float = 0.5,
) -> bool:
    """Poll a device-side file until `pattern` appears in its content.

    Returns True when the pattern is found, False when `timeout_s` elapses.
    Uses `adb shell cat <remote_path>` on each poll cycle so it works even
    for files written by nohup/background processes that may not flush
    immediately (line-buffered output eventually lands here).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc, stdout, _stderr = remote.exec(
            f"cat {shlex.quote(remote_path)} 2>/dev/null || true",
            timeout=10.0,
        )
        if rc == 0 and pattern in (stdout or ""):
            return True
        time.sleep(poll_s)
    return False


def _run_scenario_adb_signal(
    spec: ScenarioSpec,
    device_cfg: Dict[str, Any],
    model_key: str,
    out_dir: Path,
    run_idx: int,
) -> Dict[str, Any]:
    """ADB (S25/Pixel) variant for layer=signal: run real llm_manager on device,
    inject SystemSignal JSONL via ExternalMonitor over an `adb forward` tunnel.

    Engine <-> manager transport: tcp:127.0.0.1:9101 (on device).
    External monitor transport:   tcp:127.0.0.1:9102 (on device), forwarded
                                  to host local port 19102 via adb forward.
    """
    import subprocess

    device_key = device_cfg.get("_key", "galaxy_s25")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_toml = load_models_toml()
    model_entry = resolve_model_entry(models_toml, device_key, model_key)
    remote_model_path = model_entry["model_path"]
    tokenizer_path = Path(model_entry["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = PROJECT_ROOT / tokenizer_path

    conn_cfg = dict(device_cfg.get("connection", {}) or {})
    serial = resolve_adb_serial(conn_cfg.get("serial", "") or "")
    conn_cfg["serial"] = serial
    remote = adb_remote_from_dict(conn_cfg)

    generate_bin = _remote_binary_path(device_cfg, "generate")
    manager_bin = _remote_binary_path(device_cfg, "llm_manager")

    remote_work_root = device_cfg.get("paths", {}).get("work_dir", "/data/local/tmp") \
        + "/verify_runs"
    remote_run_dir = f"{remote_work_root}/{spec.id}_r{run_idx}_{os.getpid()}"
    remote.exec(f"mkdir -p {shlex.quote(remote_run_dir)}", timeout=15.0)

    work_dir = device_cfg.get("paths", {}).get("work_dir", "/data/local/tmp")
    remote.exec(
        "pkill -TERM -f 'llm_manager' 2>/dev/null; "
        "pkill -TERM -f 'mock_manager' 2>/dev/null; "
        f"pkill -TERM -f '{work_dir}/generate' 2>/dev/null; "
        "sleep 0.3; "
        "pkill -KILL -f 'llm_manager' 2>/dev/null; "
        "pkill -KILL -f 'mock_manager' 2>/dev/null; "
        f"pkill -KILL -f '{work_dir}/generate' 2>/dev/null; "
        "true",
        timeout=20.0,
    )

    backend = _resolve_backend(spec.baseline.backend, device_cfg)

    lib_dir = device_cfg.get("paths", {}).get("lib_dir", "")
    env_vars: Dict[str, str] = {"RAYON_NUM_THREADS": "6"}
    if lib_dir:
        env_vars["LD_LIBRARY_PATH"] = lib_dir

    forced_thread_args: list = []
    scenario_extra = list(spec.baseline.extra_args)
    if "--threads" not in scenario_extra:
        forced_thread_args = ["--threads", "6"]

    prompt_text = load_prompt(spec.baseline.prompt)
    local_prompt = out_dir / "prompt.txt"
    local_prompt.write_text(prompt_text, encoding="utf-8")
    remote_prompt = f"{remote_run_dir}/prompt.txt"
    remote.push(local_prompt, remote_prompt, retries=2, timeout=60.0)

    # Manager config (all monitors off except external tcp:127.0.0.1:9102).
    cfg_src = PROJECT_ROOT / "verify" / "fixtures" / "manager_config_external_only.toml"
    local_cfg = out_dir / "manager_config.toml"
    local_cfg.write_text(cfg_src.read_text(), encoding="utf-8")
    remote_cfg = f"{remote_run_dir}/manager_config.toml"
    remote.push(local_cfg, remote_cfg, retries=2, timeout=30.0)

    # Policy script (Lua) — llm_manager requires --policy-script when built
    # with the `lua` feature but without `hierarchical`.
    policy_src = PROJECT_ROOT / "manager" / "scripts" / "policy_default.lua"
    local_policy = out_dir / "policy.lua"
    local_policy.write_text(policy_src.read_text(), encoding="utf-8")
    remote_policy = f"{remote_run_dir}/policy.lua"
    remote.push(local_policy, remote_policy, retries=2, timeout=30.0)

    baseline_jsonl_local = out_dir / "baseline.jsonl"
    baseline_stdout_local = out_dir / "baseline.stdout"
    baseline_stderr_local = out_dir / "baseline.stderr"
    action_jsonl_local = out_dir / "action.jsonl"
    action_stdout_local = out_dir / "action.stdout"
    action_stderr_local = out_dir / "action.stderr"
    manager_stdout_local = out_dir / "manager.stdout"
    manager_stderr_local = out_dir / "manager.stderr"
    baseline_adb_stderr = out_dir / "baseline.adb.stderr"
    action_adb_stderr = out_dir / "action.adb.stderr"
    manager_adb_stderr = out_dir / "manager.adb.stderr"
    signal_log_local = out_dir / "signal_client.log"
    schedule_local = out_dir / "injection_schedule.json"
    verdict_path = out_dir / "verdict.json"

    baseline_jsonl_remote = f"{remote_run_dir}/baseline.jsonl"
    baseline_stdout_remote = f"{remote_run_dir}/baseline.stdout"
    baseline_stderr_remote = f"{remote_run_dir}/baseline.stderr"
    action_jsonl_remote = f"{remote_run_dir}/action.jsonl"
    action_stdout_remote = f"{remote_run_dir}/action.stdout"
    action_stderr_remote = f"{remote_run_dir}/action.stderr"
    manager_stdout_remote = f"{remote_run_dir}/manager.stdout"
    manager_pidfile_remote = f"{remote_run_dir}/llm_manager.pid"

    timings: Dict[str, float] = {}

    # ── Step 1: baseline (identical to engine_cmd adb) ───
    baseline_cmd = _generate_cmd(
        generate_bin=generate_bin,
        model_path=remote_model_path,
        prompt_file=remote_prompt,
        experiment_output=baseline_jsonl_remote,
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=False,
        resilience_transport=None,
        extra_args=scenario_extra + forced_thread_args,
    )
    t0 = time.monotonic()
    rc_b, timed_b = run_foreground_adb(
        remote=remote, cmd=baseline_cmd,
        local_stdout_path=baseline_stdout_local,
        local_stderr_path=baseline_stderr_local,
        adb_stderr_path=baseline_adb_stderr,
        remote_log_stdout=baseline_stdout_remote,
        remote_log_stderr=baseline_stderr_remote,
        env_vars=env_vars, timeout_s=900.0,
    )
    baseline_wall = time.monotonic() - t0
    timings["baseline_wall_s"] = baseline_wall

    try:
        remote.pull(baseline_jsonl_remote, baseline_jsonl_local, retries=2, timeout=120.0)
    except Exception as e:
        print(f"  [warn] baseline.jsonl pull failed: {e}")
        if not baseline_jsonl_local.exists():
            baseline_jsonl_local.write_bytes(b"")

    # ── Step 2: action (llm_manager + engine + signal inject) ───
    schedule = list(spec.action.injection_schedule or [])
    if not schedule:
        raise ValueError(
            f"signal scenario {spec.id} requires action.injection_schedule"
        )
    schedule_local.write_text(json.dumps(schedule, indent=2), encoding="utf-8")

    engine_transport = "tcp:127.0.0.1:9101"
    # device-side signal endpoint that manager's ExternalMonitor will bind
    ext_monitor_device_port = 9102
    host_forward_port = 19102

    mgr_cmd = [
        manager_bin,
        "--transport", "tcp:127.0.0.1:9101",
        "--config", remote_cfg,
        "--policy-script", remote_policy,
    ]
    mgr_env = dict(env_vars)
    mgr_env["RUST_LOG"] = "info"

    rc_a: Optional[int] = None
    timed_a = False
    total_delay = sum(float(e.get("delay_sec", 0.0)) for e in schedule)
    action_timeout = max(360.0, baseline_wall * 2.0 + total_delay + 120.0)

    action_cmd = _generate_cmd(
        generate_bin=generate_bin,
        model_path=remote_model_path,
        prompt_file=remote_prompt,
        experiment_output=action_jsonl_remote,
        decode_tokens=spec.baseline.decode_tokens,
        prefill_chunk_size=spec.baseline.prefill_chunk_size,
        backend=backend,
        enable_resilience=True,
        resilience_transport=engine_transport,
        extra_args=scenario_extra + forced_thread_args,
    )

    # action_rc_remote: device-side file where engine exit code is written.
    # Engine runs in background via AdbBackgroundProcess-like wrapper so that
    # we can interleave the ExternalMonitor bind wait and signal_client start.
    action_pidfile_remote = f"{remote_run_dir}/engine.pid"
    action_rc_remote = f"{remote_run_dir}/engine.rc"

    t1 = time.monotonic()
    signal_proc = None
    try:
        with AdbBackgroundProcess(
            remote=remote,
            cmd=mgr_cmd,
            remote_log=manager_stdout_remote,
            remote_pidfile=manager_pidfile_remote,
            adb_stderr_log=manager_adb_stderr,
            env_vars=mgr_env,
            startup_wait_s=0.5,
        ) as _mgr_proc:
            # Give llm_manager time to bind the engine-facing TCP listener
            # (port 9101). The ExternalMonitor port (9102) is bound only AFTER
            # the engine client connects, so we cannot adb-forward yet.
            time.sleep(2.0)

            # Tunnel host → device for the external-monitor port.
            _adb_forward(host_forward_port, ext_monitor_device_port, serial)

            # Start the engine in the background. The wrapper shell captures
            # the exit code to action_rc_remote so we can detect engine
            # completion without blocking.
            env_prefix = AdbRemote._inline_env(env_vars)
            engine_cmd_str = " ".join(shlex.quote(a) for a in action_cmd)
            # Use the same detach pattern as AdbBackgroundProcess (nohup +
            # redirect + disown + explicit exit 0) so that `adb shell`
            # returns immediately instead of blocking on the backgrounded
            # child's stdio. Without this, the adb session waits until the
            # child closes stdout/stderr, causing remote.exec to timeout
            # after 15s (observed on S25).
            engine_script = (
                f"{env_prefix}nohup sh -c "
                + shlex.quote(
                    f"{engine_cmd_str} < /dev/null "
                    f"> {shlex.quote(action_stdout_remote)} "
                    f"2> {shlex.quote(action_stderr_remote)}; "
                    f"echo $? > {shlex.quote(action_rc_remote)}"
                )
                + f" > /dev/null 2>&1 < /dev/null & "
                f"echo $! > {shlex.quote(action_pidfile_remote)}; "
                "disown 2>/dev/null || true; "
                "exit 0"
            )
            remote.exec(
                f"sh -c {shlex.quote(engine_script)}",
                timeout=15.0,
            )

            # Fixed sleep to let engine finish model load and handshake with
            # manager. Empirical: S25 F16 model load takes ~9s. Engine
            # connects to manager's 9101 right after load; ExternalMonitor
            # binds immediately afterwards. signal_client must start only
            # AFTER this handshake — otherwise its connect hits a not-yet-
            # bound port, and adb forward may accept TCP without routing
            # bytes (losing the signal). 10s gives ~1s margin.
            time.sleep(10.0)

            # Start signal_client AFTER the engine is confirmed connected
            # (ExternalMonitor is now bound). signal_client's internal 30s
            # retry absorbs any residual timing slack.
            sig_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "verify" / "harness" / "signal_client.py"),
                "--transport", f"tcp:127.0.0.1:{host_forward_port}",
                "--schedule", str(schedule_local),
                "--log-file", str(signal_log_local),
                "--connect-timeout", "30",
                "--pre-sleep", "0",
            ]
            signal_proc = subprocess.Popen(
                sig_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_ROOT),
            )

            # Wait for the engine background process to finish.
            # Poll the rc file; fall back to action_timeout.
            deadline_engine = time.monotonic() + action_timeout
            while time.monotonic() < deadline_engine:
                rc_chk, rc_out, _ = remote.exec(
                    f"cat {shlex.quote(action_rc_remote)} 2>/dev/null || true",
                    timeout=10.0,
                )
                if rc_chk == 0 and (rc_out or "").strip().isdigit():
                    rc_a = int(rc_out.strip())
                    break
                time.sleep(1.0)
            else:
                timed_a = True
                rc_a = -1
                # Kill the engine process if it's still alive.
                rc_pid, pid_out, _ = remote.exec(
                    f"cat {shlex.quote(action_pidfile_remote)} 2>/dev/null || true",
                    timeout=10.0,
                )
                if rc_pid == 0 and (pid_out or "").strip().isdigit():
                    engine_pid = int(pid_out.strip())
                    remote.exec(
                        f"kill -TERM {engine_pid} 2>/dev/null; "
                        f"sleep 1; kill -KILL {engine_pid} 2>/dev/null; true",
                        timeout=15.0,
                    )

            # Pull engine stdout/stderr now that the process has exited.
            try:
                remote.pull(action_stdout_remote, action_stdout_local, retries=2, timeout=120.0)
            except Exception as e:
                print(f"  [warn] action.stdout pull failed: {e}")
                if not action_stdout_local.exists():
                    action_stdout_local.write_bytes(b"")
            try:
                remote.pull(action_stderr_remote, action_stderr_local, retries=2, timeout=120.0)
            except Exception as e:
                print(f"  [warn] action.stderr pull failed: {e}")
                if not action_stderr_local.exists():
                    action_stderr_local.write_bytes(b"")

            time.sleep(1.0)
    finally:
        if signal_proc is not None:
            try:
                signal_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                signal_proc.kill()
        _adb_forward_remove(host_forward_port, serial)

    try:
        remote.pull(manager_stdout_remote, manager_stdout_local, retries=2, timeout=60.0)
    except Exception as e:
        print(f"  [warn] manager.stdout pull failed: {e}")
        if not manager_stdout_local.exists():
            manager_stdout_local.write_bytes(b"")

    try:
        remote.pull(action_jsonl_remote, action_jsonl_local, retries=2, timeout=120.0)
    except Exception as e:
        print(f"  [warn] action.jsonl pull failed: {e}")
        if not action_jsonl_local.exists():
            action_jsonl_local.write_bytes(b"")

    # action.stdout / action.stderr were pulled inside the manager context
    # block above (right after engine exit). Ensure fallback files exist in
    # case the inner pull was skipped due to an early exception.
    if not action_stdout_local.exists():
        action_stdout_local.write_bytes(b"")
    if not action_stderr_local.exists():
        action_stderr_local.write_bytes(b"")

    if not manager_stderr_local.exists():
        manager_stderr_local.write_bytes(b"")

    timings["action_wall_s"] = time.monotonic() - t1

    return _finalize_verdict(
        spec=spec, device_key=device_key, model_key=model_key, run_idx=run_idx,
        backend=backend, use_scenario_mode=False, scenario_file=None,
        timings=timings,
        baseline_jsonl=baseline_jsonl_local, action_jsonl=action_jsonl_local,
        action_stderr=action_stderr_local, manager_stdout=manager_stdout_local,
        out_dir=out_dir,
        rc_b=rc_b, timed_b=timed_b, rc_a=rc_a, timed_a=timed_a,
        tokenizer_path=tokenizer_path, verdict_path=verdict_path,
    )


# ── Shared tail: detokenize + assertions + verdict ────


def _finalize_verdict(
    *,
    spec: ScenarioSpec,
    device_key: str,
    model_key: str,
    run_idx: int,
    backend: str,
    use_scenario_mode: bool,
    scenario_file: Optional[Path],
    timings: Dict[str, float],
    baseline_jsonl: Path,
    action_jsonl: Path,
    action_stderr: Path,
    manager_stdout: Path,
    out_dir: Path,
    rc_b: int,
    timed_b: bool,
    rc_a: Optional[int],
    timed_a: bool,
    tokenizer_path: Path,
    verdict_path: Path,
) -> Dict[str, Any]:

    # Detokenize
    try:
        baseline_text = decode_jsonl_to_text(baseline_jsonl, tokenizer_path)
    except Exception as e:
        baseline_text = ""
        (out_dir / "baseline.decode_error.txt").write_text(repr(e))
    try:
        action_text = decode_jsonl_to_text(action_jsonl, tokenizer_path)
    except Exception as e:
        action_text = ""
        (out_dir / "action.decode_error.txt").write_text(repr(e))

    (out_dir / "baseline.decoded.txt").write_text(baseline_text, encoding="utf-8")
    (out_dir / "action.decoded.txt").write_text(action_text, encoding="utf-8")

    # Parse
    baseline_summary = load_summary(baseline_jsonl)
    action_summary = load_summary(action_jsonl)
    heartbeats = parse_heartbeats(manager_stdout)
    (out_dir / "heartbeats.json").write_text(json.dumps(heartbeats, indent=2))

    # Verify
    functional = verify_functional(spec.expected.functional, action_stderr, heartbeats)
    performance = verify_performance(
        spec.expected.performance, baseline_summary, action_summary
    )
    accuracy = verify_accuracy(spec.expected.accuracy, baseline_text, action_text)
    crash_progress = verify_crash_and_progress(
        spec.expected.crash_and_progress,
        action_stderr,
        action_jsonl,
        int(spec.baseline.decode_tokens),
        rc_a,
    )

    verdict_sub = aggregate_verdict(
        functional, performance, accuracy, spec.pass_criteria,
        crash_and_progress=crash_progress,
    )

    verdict = {
        "scenario_id": spec.id,
        "device": device_key,
        "model": model_key,
        "run_idx": run_idx,
        "backend": backend,
        "layer": spec.layer,
        "mode": "scenario" if use_scenario_mode else "single",
        "scenario_file": str(scenario_file) if scenario_file else None,
        "timings": timings,
        "baseline_returncode": rc_b,
        "baseline_timed_out": timed_b,
        "action_returncode": rc_a,
        "action_timed_out": timed_a,
        "functional": functional,
        "performance": performance,
        "accuracy": accuracy,
        "crash_and_progress": crash_progress,
        "overall_pass": verdict_sub["overall_pass"],
        "pass_criteria": spec.pass_criteria,
        "heartbeat_count": len(heartbeats),
    }
    verdict_path.write_text(json.dumps(verdict, indent=2))
    return verdict
