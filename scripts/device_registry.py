#!/usr/bin/env python3
"""Device Registry CLI — discover, list, and validate devices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path for package imports
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from device_registry.config import find_devices_toml, load_all_devices
from device_registry.discover import discover_all


def _format_toml_value(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, int):
        return str(v)
    elif isinstance(v, list):
        items = ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in v)
        return f"[{items}]"
    return f'"{v}"'


def _dict_to_toml(device_id: str, d: dict, indent: str = "") -> str:
    """Convert a device dict to TOML string."""
    lines = []
    prefix = f"devices.{device_id}"

    # Top-level scalars
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{indent}{k} = {_format_toml_value(v)}")

    # Sub-tables
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"\n[{prefix}.{k}]")
            for sk, sv in v.items():
                lines.append(f"{sk} = {_format_toml_value(sv)}")

    return "\n".join(lines)


def cmd_discover(args: argparse.Namespace) -> int:
    """Discover devices and optionally register them."""
    devices = discover_all()

    if not devices:
        print("No devices discovered.")
        return 1

    print(f"\n{'='*50}")
    print(f" Discovered Devices")
    print(f"{'='*50}")
    for i, dev in enumerate(devices, 1):
        info_parts = [dev.arch]
        if dev.memory_gb:
            info_parts.append(f"{dev.memory_gb}GB")
        if dev.gpu:
            info_parts.append(dev.gpu)
        info = ", ".join(info_parts)

        type_label = dev.connection_type.upper()
        serial_suffix = f" [{dev.serial}]" if dev.serial else ""
        print(f" [{i}] {type_label}: {dev.name}{serial_suffix} ({info})")
    print(f"{'='*50}\n")

    # Interactive registration
    registrations: dict[str, dict] = {}
    for i, dev in enumerate(devices, 1):
        try:
            answer = input(
                f"Register [{i}] as? ({dev.suggested_id}) [Enter=accept, s=skip] > "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1

        if answer.lower() == "s":
            continue
        device_id = answer if answer else dev.suggested_id
        registrations[device_id] = dev.to_toml_dict()
        print(f"  -> Registered as '{device_id}'")

    if not registrations:
        print("No devices registered.")
        return 0

    # Generate TOML
    toml_sections = []
    for dev_id, dev_dict in registrations.items():
        toml_sections.append(f"[devices.{dev_id}]")
        toml_sections.append(_dict_to_toml(dev_id, dev_dict))

    toml_content = "\n\n".join(toml_sections) + "\n"

    # Write
    project_root = _SCRIPTS_DIR.parent
    toml_path = project_root / "devices.toml"

    if args.append and toml_path.exists():
        with open(toml_path, "a") as f:
            f.write("\n" + toml_content)
        print(f"\nAppended to {toml_path}")
    else:
        if toml_path.exists() and not args.append:
            try:
                confirm = input(
                    f"{toml_path} exists. Overwrite? [y/N] > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 1
            if confirm != "y":
                print("Aborted.")
                return 0
        with open(toml_path, "w") as f:
            f.write(toml_content)
        print(f"\nWritten to {toml_path}")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List registered devices."""
    try:
        devices = load_all_devices()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not devices:
        print("No devices registered in devices.toml")
        return 0

    print(f"\n{'ID':<15} {'Name':<25} {'Type':<8} {'Target':<30} {'Work Dir'}")
    print("-" * 95)
    for dev_id, cfg in devices.items():
        target = cfg.build.target or "(native)"
        print(
            f"{dev_id:<15} {cfg.name:<25} {cfg.connection.type:<8} "
            f"{target:<30} {cfg.paths.work_dir}"
        )
    print()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate devices.toml."""
    try:
        toml_path = find_devices_toml()
    except FileNotFoundError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"Validating {toml_path}...")
    errors = []

    try:
        devices = load_all_devices(toml_path)
    except Exception as e:
        print(f"FAIL: Parse error: {e}", file=sys.stderr)
        return 1

    for dev_id, cfg in devices.items():
        if cfg.connection.type not in ("local", "adb", "ssh"):
            errors.append(f"[{dev_id}] Unknown connection type: {cfg.connection.type}")
        if not cfg.paths.work_dir:
            errors.append(f"[{dev_id}] Missing paths.work_dir")
        if cfg.connection.type == "adb" and not cfg.build.target:
            errors.append(f"[{dev_id}] ADB device should have build.target set")

    if errors:
        print(f"FAIL: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK: {len(devices)} device(s) validated.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Device Registry — manage device configurations"
    )
    sub = parser.add_subparsers(dest="command")

    # discover
    p_disc = sub.add_parser("discover", help="Discover and register devices")
    p_disc.add_argument(
        "--append", action="store_true", help="Append to existing devices.toml"
    )

    # list
    sub.add_parser("list", help="List registered devices")

    # validate
    sub.add_parser("validate", help="Validate devices.toml")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "discover": cmd_discover,
        "list": cmd_list,
        "validate": cmd_validate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
