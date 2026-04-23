#!/usr/bin/env python3
"""Resilience verify harness — CLI entry point.

Typical usage (run from project root):

    python verify/verify.py --device host --model f16 \\
        --scenario-filter throttle_smoke --runs 1 --skip-deploy

    python verify/verify.py --device galaxy_s25 --model f16,q4 --runs 1

See verify/USAGE.md for the full CLI reference and YAML scenario
authoring guide; verify/README.md for architecture overview.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make harness/ importable when invoked as a plain script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from harness.fixtures import (  # noqa: E402
    PROJECT_ROOT,
    RESULTS_DIR,
    SCENARIOS_DIR,
    load_device_config,
)
from harness.orchestrator import run_scenario  # noqa: E402
from harness.report import (  # noqa: E402
    render_console_table,
    write_summary_jsonl,
    write_summary_md,
)
from harness.spec_loader import (  # noqa: E402
    discover_scenarios,
    filter_scenarios,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Resilience verify harness")
    p.add_argument("--device", default="host", help="Device key from devices.toml")
    p.add_argument(
        "--model",
        default="f16",
        help="Comma-separated model keys (e.g. 'f16' or 'f16,q4')",
    )
    p.add_argument("--scenario-filter", default=None)
    p.add_argument(
        "--layer",
        default="both",
        help="signal | engine_cmd | both (default: both)",
    )
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-deploy", action="store_true")
    p.add_argument(
        "--output-root",
        default=str(RESULTS_DIR),
        help="Root directory for result artifacts",
    )
    p.add_argument(
        "--scenarios-dir",
        default=str(SCENARIOS_DIR),
        help="Directory to scan for YAML scenarios",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


# ── Build & deploy (remote devices only) ────────────────────────────────


def _build_remote_binaries(device_key: str) -> bool:
    """Cross-compile generate + mock_manager for a remote device.

    Uses scripts/device_registry/builder.py::build_binary, which honours the
    zig_target / toolchain settings in devices.toml. The mock_manager needs
    the `lua` feature (which is not part of the engine crate's feature set),
    so we call the builder twice with distinct feature lists.
    """
    from device_registry.config import load_device_config as _load_dc  # type: ignore
    from device_registry.builder import build_binary  # type: ignore

    device_cfg = _load_dc(device_key)

    print(f"[verify] cross-compiling generate for {device_key} ...")
    ok_gen = build_binary(
        device_config=device_cfg,
        binary_name="generate",
        project_root=PROJECT_ROOT,
    )
    if not ok_gen:
        print("[verify] generate build FAILED")
        return False

    # mock_manager lives in the manager crate which requires the `lua` feature
    # (policy_default.lua is loaded at startup). We use an auxiliary build
    # invocation via a shallow wrapper over build_binary.
    print(f"[verify] cross-compiling mock_manager for {device_key} ...")
    ok_mock = _build_manager_binary(device_cfg, "mock_manager")
    if not ok_mock:
        print("[verify] mock_manager build FAILED")
        return False

    # llm_manager: real policy service used by layer=signal scenarios.
    print(f"[verify] cross-compiling llm_manager for {device_key} ...")
    ok_mgr = _build_manager_binary(device_cfg, "llm_manager")
    if not ok_mgr:
        print("[verify] llm_manager build FAILED")
        return False
    return True


def _build_manager_binary(device_cfg, bin_name: str) -> bool:
    """Cross-compile a llm_manager crate binary (`mock_manager` or
    `llm_manager`) with the `lua` feature.

    Composes NDK/cross toolchain env from hosts.toml when applicable,
    mirroring build_binary() but invoking cargo with --no-default-features
    --features lua --package llm_manager --bin <bin_name>.
    """
    import os as _os
    import subprocess

    from device_registry.builder import _compose_toolchain_env  # type: ignore
    from device_registry.config import (  # type: ignore
        detect_current_host,
        load_hosts_config,
    )

    use_zigbuild = bool(device_cfg.build.zig_target)
    env = dict(_os.environ)

    if use_zigbuild:
        pass
    elif device_cfg.build.toolchain:
        try:
            hosts_cfg = load_hosts_config()
            host = detect_current_host(hosts_cfg)
            toolchain_env = _compose_toolchain_env(
                host,
                device_cfg.build.toolchain,
                device_cfg.build.target or None,
            )
            env.update(toolchain_env)
        except Exception as exc:
            print(f"[verify] mock_manager toolchain env resolve failed: {exc}")
            return False

    cmd_parts: list[str] = []
    if use_zigbuild:
        cmd_parts = ["cargo", "zigbuild", "--release"]
    else:
        cmd_parts = ["cargo", "build", "--release"]

    cmd_parts.extend(["--no-default-features", "--features", "lua"])

    if use_zigbuild:
        cmd_parts.extend(["--target", device_cfg.build.zig_target])
    elif device_cfg.build.target:
        cmd_parts.extend(["--target", device_cfg.build.target])

    cmd_parts.extend(["-p", "llm_manager", "--bin", bin_name])

    cmd = " ".join(cmd_parts)
    print(f"  Build: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT), env=env)
    return r.returncode == 0


def _deploy_remote_binaries(device_key: str) -> bool:
    """Push generate + mock_manager to the remote device under paths.work_dir."""
    from device_registry.config import load_device_config as _load_dc  # type: ignore
    from device_registry.connection import create_connection  # type: ignore
    from device_registry.deployer import deploy_binary  # type: ignore
    from device_registry.builder import get_local_binary_path  # type: ignore

    device_cfg = _load_dc(device_key)
    conn = create_connection(device_cfg.connection)

    for name in ("generate", "mock_manager", "llm_manager"):
        local = get_local_binary_path(device_cfg, name, PROJECT_ROOT)
        if not local.exists():
            print(f"[verify] {local} missing; skip --skip-build and retry")
            return False
        ok = deploy_binary(
            conn=conn,
            device_config=device_cfg,
            local_binary_path=local,
            binary_name=name,
        )
        if not ok:
            print(f"[verify] deploy {name} failed")
            return False
    return True


def _deploy_prompts(device_key: str) -> bool:
    """Push fixtures/prompts/ to the remote device (small text files).

    Not strictly required because the orchestrator writes a scenario-specific
    prompt.txt to its run-dir via scp, but we keep a master copy at
    <work_dir>/verify_prompts/ for debugging.
    """
    from device_registry.config import load_device_config as _load_dc  # type: ignore
    from device_registry.connection import create_connection  # type: ignore

    device_cfg = _load_dc(device_key)
    conn = create_connection(device_cfg.connection)
    prompts_src = PROJECT_ROOT / "experiments" / "verify" / "fixtures" / "prompts"
    remote_dir = f"{device_cfg.paths.work_dir}/verify_prompts"
    conn.mkdir(remote_dir)
    for p in sorted(prompts_src.glob("*.txt")):
        conn.push(p, f"{remote_dir}/{p.name}")
    return True


# ── Main ────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    args = parse_args(argv)

    device_cfg = load_device_config(device_key=args.device)
    models = [m.strip() for m in args.model.split(",") if m.strip()]

    conn_type = device_cfg.get("connection", {}).get("type", "local")
    is_remote = conn_type in ("ssh", "adb")

    # ── Build & deploy (remote only) ───────────────────
    if is_remote:
        if not args.skip_build:
            if not _build_remote_binaries(args.device):
                print("[verify] Build step failed — aborting.")
                return 3
        if not args.skip_deploy:
            if not _deploy_remote_binaries(args.device):
                print("[verify] Deploy step failed — aborting.")
                return 3
            try:
                _deploy_prompts(args.device)
            except Exception as e:
                print(f"[verify] (non-fatal) prompt deploy failed: {e}")

    # ── Discover scenarios ─────────────────────────────
    scenarios = discover_scenarios(Path(args.scenarios_dir))
    scenarios = filter_scenarios(scenarios, args.scenario_filter)

    if args.layer not in ("both",):
        scenarios = [s for s in scenarios if s.layer == args.layer]

    scenarios = [s for s in scenarios if args.device in s.devices or not s.devices]

    plan: list = []
    for s in scenarios:
        allowed = set(s.models) if s.models else set(models)
        for m in models:
            if m in allowed:
                plan.append((s, m))

    if not plan:
        print("No scenario × model combos after filtering.")
        return 1

    if args.dry_run:
        print("DRY RUN — planned executions:")
        for s, m in plan:
            print(f"  - {s.id}  device={args.device}  model={m}  runs={args.runs}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root) / f"{timestamp}_{args.device}_{'_'.join(models)}"
    root.mkdir(parents=True, exist_ok=True)

    # Include `model` in the scenario dir so that a multi-model matrix
    # (e.g. q4,f16) doesn't have each model overwrite the previous one's
    # verdict. Single-model runs still get a clean <scenario>/r<idx> tree
    # when model is the sole key.
    multi_model = len(models) > 1

    all_verdicts: list = []
    for spec, model in plan:
        for run_idx in range(args.runs):
            if multi_model:
                scen_dir = root / spec.id / model / f"r{run_idx}"
            else:
                scen_dir = root / spec.id / f"r{run_idx}"
            print(f"[verify] Running {spec.id} on {args.device}/{model} run={run_idx}")
            try:
                verdict = run_scenario(
                    spec=spec,
                    device_cfg=device_cfg,
                    model_key=model,
                    out_dir=scen_dir,
                    run_idx=run_idx,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                verdict = {
                    "scenario_id": spec.id,
                    "device": args.device,
                    "model": model,
                    "run_idx": run_idx,
                    "error": repr(e),
                    "overall_pass": False,
                }
            all_verdicts.append(verdict)
            status = "PASS" if verdict.get("overall_pass") else "FAIL"
            print(f"  → {status}")

    summary_md = root / "summary.md"
    summary_jsonl = root / "summary.jsonl"
    write_summary_md(all_verdicts, summary_md)
    write_summary_jsonl(all_verdicts, summary_jsonl)

    print("")
    print(render_console_table(all_verdicts))
    print("")
    print(f"Summary markdown: {summary_md}")
    print(f"Summary jsonl:    {summary_jsonl}")

    all_pass = all(v.get("overall_pass") for v in all_verdicts)
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
