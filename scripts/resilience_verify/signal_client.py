"""SystemSignal JSON-Lines injector for llm_manager ExternalMonitor.

Connects to a TCP endpoint (preferred for remote/Android SELinux reasons) or
Unix domain socket and writes one SystemSignal JSON per line, separated by
newline. The schedule describes relative delays in seconds between signals.

Schedule file format:
    [
      {"delay_sec": 0.5, "signal": {"memory_pressure": {...}}},
      {"delay_sec": 2.0, "signal": {"thermal_alert":   {...}}}
    ]

Usage:
    signal_client.py --transport tcp:127.0.0.1:9102 --schedule schedule.json
    signal_client.py --transport unix:/tmp/sig.sock   --schedule schedule.json

The client retries connect for up to --connect-timeout seconds (default 15)
to give the manager time to bind its external monitor listener. Broken pipes
are logged and treated as non-fatal (scenario may proceed independently).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _connect_tcp(addr: str, timeout_s: float) -> socket.socket:
    host, _, port_s = addr.partition(":")
    port = int(port_s)
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2.0)
            s.settimeout(None)
            return s
        except OSError as e:
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"tcp connect to {addr} timed out: {last_err}")


def _connect_unix(path: str, timeout_s: float) -> socket.socket:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(path)
            return s
        except OSError as e:
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"unix connect to {path} timed out: {last_err}")


def connect(transport: str, timeout_s: float) -> socket.socket:
    if transport.startswith("tcp:"):
        return _connect_tcp(transport[4:], timeout_s)
    if transport.startswith("unix:"):
        return _connect_unix(transport[5:], timeout_s)
    raise ValueError(f"unsupported transport: {transport}")


def send_schedule(sock: socket.socket, schedule: List[Dict[str, Any]], log=print) -> None:
    for idx, entry in enumerate(schedule):
        delay = float(entry.get("delay_sec", 0.0))
        if delay > 0:
            time.sleep(delay)
        sig = entry.get("signal")
        if not isinstance(sig, dict):
            log(f"[signal_client] skipping entry {idx}: missing or malformed 'signal'")
            continue
        line = json.dumps(sig, separators=(",", ":")) + "\n"
        try:
            sock.sendall(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError) as e:
            log(f"[signal_client] broken pipe at entry {idx}: {e} — stopping")
            return
        log(f"[signal_client] [{idx + 1}/{len(schedule)}] sent {list(sig.keys())[0] if sig else '?'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", required=True, help="tcp:<host>:<port> or unix:<path>")
    p.add_argument("--schedule", required=True, type=Path, help="JSON schedule file")
    p.add_argument("--connect-timeout", type=float, default=15.0)
    p.add_argument(
        "--pre-sleep",
        type=float,
        default=0.0,
        help="Sleep N seconds BEFORE first connect attempt. Useful when the "
             "remote listener (e.g. ExternalMonitor) hasn't bound yet.",
    )
    p.add_argument("--log-file", type=Path, default=None, help="append log lines here (else stderr)")
    args = p.parse_args()

    schedule = json.loads(args.schedule.read_text())
    if not isinstance(schedule, list):
        print("[signal_client] schedule must be a JSON array", file=sys.stderr)
        return 2

    log_fn = print
    log_fh = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log_file, "a", encoding="utf-8")

        def _log(msg: str) -> None:
            log_fh.write(msg + "\n")
            log_fh.flush()

        log_fn = _log

    try:
        if args.pre_sleep > 0:
            log_fn(f"[signal_client] pre-sleep {args.pre_sleep}s before connect")
            time.sleep(args.pre_sleep)
        sock = connect(args.transport, args.connect_timeout)
        log_fn(f"[signal_client] connected to {args.transport}")
        send_schedule(sock, schedule, log=log_fn)
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        sock.close()
        log_fn("[signal_client] done")
        return 0
    finally:
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
