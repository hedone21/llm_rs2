#!/usr/bin/env python3
"""microbench_qnn_matrix.py — QNN HTP 비교 측정 driver.

9-cell + Ref 매트릭스를 신뢰성 있게 측정.

핵심 protocol (plan_qnn_microbench_measurement_protocol_2026_05_26.md):
- warmup 3 + measure 10 trial per cell
- round-robin shuffle (cell 순서 발열 누적 편향 제거)
- 8 thermal zone (CPU little/mid/prime + GPU + Hexagon HVX/HMX + DDR) polling
- 50°C trigger / 60s session warmup / 45~180s inter-cell cooldown
- Tukey 1.5×IQR outlier rejection
- raw + aggregated + thermal_log + report.md 출력

S25 zone naming (Phase A.2 결과):
- nsphvx-* (zone 39~41): Hexagon HVX (vector)
- nsphmx-* (zone 42~45): Hexagon HMX (matrix, HTP 핵심)
- gpuss-* (zone 23~30): Adreno
- cpu-0-*/cpu-1-*: CPU clusters
- ddr (zone 46)

bench_strict_thermal_isolation.sh 의 thermal logic 을 Python 으로 포팅.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
# devices.toml: galaxy_s25.paths.work_dir = /data/local/tmp (run_device.py push 위치).
DEVICE_BIN_DIR = "/data/local/tmp"
DEVICE_QNN_LIBDIR = "/data/local/tmp/qnn"
DEVICE_WORK_DIR = "/data/local/tmp"

# ----------------------------------------------------------------------
# Thermal monitoring (S25 SM8750, Phase A.2 결과 반영)
# ----------------------------------------------------------------------

ZONES: Dict[str, int] = {
    "cpu_little": 1,    # cpu-0-0-0
    "cpu_mid":    9,    # cpu-0-4-0
    "cpu_prime":  16,   # cpu-1-0-0
    "gpu_5":      28,   # gpuss-5
    "gpu_7":      30,   # gpuss-7
    "hex_vec":    39,   # nsphvx-0 (Hexagon Vector)
    "hex_mat":    42,   # nsphmx-0 (Hexagon Matrix, HTP 핵심)
    "ddr":        46,   # ddr
}

TRIGGER_TEMP_C = 50.0
RECOVERY_TEMP_C = 38.0
DANGER_TEMP_C = 60.0
START_TEMP_C = 40.0

SESSION_WARMUP_S = 60
COOLDOWN_MIN_S = 45
COOLDOWN_MAX_S = 180
COOLDOWN_TRIGGER_S = 600  # 50°C 도달 시 회복 wait
INTER_TRIAL_S = 5
INTER_TRIAL_MAX_S = 30
INTER_ROUND_S = 300
INTER_ROUND_MAX_S = 600
THROTTLING_RATIO = 1.15

WARMUP_TRIALS = 3
MEASURE_TRIALS = 10
ROUNDS = WARMUP_TRIALS + MEASURE_TRIALS  # 13


# ----------------------------------------------------------------------
# ADB wrapper
# ----------------------------------------------------------------------

class Adb:
    def __init__(self, serial: str):
        self.serial = serial

    def shell(self, cmd: str, timeout: int = 120, check: bool = True) -> str:
        p = subprocess.run(
            ["adb", "-s", self.serial, "shell", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if check and p.returncode != 0:
            raise RuntimeError(
                f"adb shell failed (rc={p.returncode}): {cmd!r}\nstderr={p.stderr}"
            )
        return p.stdout.rstrip("\r\n")

    def push(self, src: str, dst: str) -> None:
        subprocess.run(
            ["adb", "-s", self.serial, "push", src, dst],
            check=True, capture_output=True, text=True,
        )

    def read_zones(self) -> Dict[str, float]:
        paths = " ".join(f"/sys/class/thermal/thermal_zone{z}/temp" for z in ZONES.values())
        raw = self.shell(f"cat {paths}", timeout=10)
        temps = [int(x.strip()) for x in raw.split() if x.strip().lstrip("-").isdigit()]
        if len(temps) != len(ZONES):
            raise RuntimeError(f"zone count mismatch: {len(temps)} vs {len(ZONES)}")
        return {name: temps[i] / 1000.0 for i, name in enumerate(ZONES.keys())}

    def max_zone_temp(self) -> Tuple[str, float]:
        temps = self.read_zones()
        name = max(temps, key=temps.get)
        return name, temps[name]

    def zombie_check(self) -> List[str]:
        out = self.shell(
            "ps -A 2>/dev/null | grep -E '(microbench|generate|qnn_executor|llama-cli)' | grep -v grep",
            check=False,
        )
        return [l.strip() for l in out.splitlines() if l.strip()]


# ----------------------------------------------------------------------
# Cell registry
# ----------------------------------------------------------------------

K = 1536
N = 8960

@dataclass
class Cell:
    name: str
    dtype: str
    backend: str
    bin_name: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    tolerance_max_abs: float = 1e-3
    tolerance_cosine: float = 0.999
    enabled: bool = True
    optional: bool = False
    # 같은 group_key 의 cell 은 한 번만 bin 실행, 결과 stdout 을 모든 cell 의 parser 가 공유.
    # 예: M3/M4 둘 다 qnngpu_matmul_tbt 의 stdout 에서 각자 다른 line 추출.
    group_key: Optional[str] = None
    latency_pattern: Optional[str] = None  # regex with 1 capture group (ms)
    # cell 별 path override (Executorch path = /data/local/tmp/executorch 등)
    work_dir_override: Optional[str] = None
    bin_dir_override: Optional[str] = None
    extra_ld_paths: List[str] = field(default_factory=list)

    def adb_cmd(self, work_dir: str) -> str:
        env_str = " ".join(f"{k}={v}" for k, v in self.env.items())
        env_prefix = f"{env_str} " if env_str else ""
        bin_dir = self.bin_dir_override or DEVICE_BIN_DIR
        bin_path = f"{bin_dir}/{self.bin_name}"
        args_str = " ".join(self.args)
        wd = self.work_dir_override or work_dir
        # LD_LIBRARY_PATH: work_dir + extra + QNN libs + system
        ld_parts = [wd] + self.extra_ld_paths + [DEVICE_QNN_LIBDIR, "/system/lib64"]
        ld_path = ":".join(ld_parts)
        adsp = f"ADSP_LIBRARY_PATH={wd} "
        return (
            f"cd {wd} && "
            f"LD_LIBRARY_PATH={ld_path} {adsp}{env_prefix}"
            f"taskset 3f {bin_path} {args_str}"
        )


def build_cells(enable: List[str]) -> List[Cell]:
    # NOTE: 각 bin 의 CLI 형식이 다양함. 실제 source 확인 결과:
    # - htp_matmul_correctness: positional `<K> <N>` (default 1024×4096)
    # - oppkg_gemv_vs_baseline: positional `<n_iters>`, K/N hardcoded 1536×8960
    # - qnn_oppkg_matmul_q40_correct: env var QNN_BACKEND_LIB, K/N hardcoded
    #   (한 bin 에서 baseline+OpPackage 둘 다 측정 → M3/M4 결합)
    # - htp_matmul_w8a8: 미존재, Phase B M2 신규 작성
    # - qnn_executor_runner: Phase D 진입 후 결정
    n_iters_str = str(MEASURE_TRIALS)
    cells = [
        Cell("M1",  "fp32", "htp",
             "microbench_htp_matmul_correctness",
             args=[str(K), str(N)],  # positional
             tolerance_max_abs=1e-3, tolerance_cosine=0.9999),
        Cell("M1b", "f16",  "htp",
             "microbench_htp_matmul_correctness",
             args=[str(K), str(N), "f16"],  # 신규 dtype arg (Phase B.M1b)
             tolerance_max_abs=1e-2, tolerance_cosine=0.999,
             optional=True),
        Cell("M2",  "w8a8", "htp",
             "microbench_htp_matmul_w8a8",
             args=[str(K), str(N)],
             tolerance_max_abs=0.1, tolerance_cosine=0.99,
             optional=True),
        # M3/M4: qnngpu_matmul_tbt 한 bin 의 dual-result.
        # output:
        #   "Baseline OpenCL mul_mv_f16_f32 ... median=X ms"
        #   "Test     QNN-GPU MAT_MUL ... median=Y ms"
        # 같은 bin run → group_key 로 stdout 공유. latency_pattern 으로 cell 별 추출.
        Cell("M3",  "f16",  "opencl",
             "microbench_qnngpu_matmul_tbt",
             args=[str(K), str(N), n_iters_str],
             tolerance_max_abs=1e-2, tolerance_cosine=0.999,
             group_key="qnngpu_tbt",
             latency_pattern=r"Baseline\s+OpenCL.*median=([\d.]+)\s*ms"),
        Cell("M4",  "f16",  "qnn-gpu",
             "microbench_qnngpu_matmul_tbt",
             args=[str(K), str(N), n_iters_str],
             tolerance_max_abs=1e-2, tolerance_cosine=0.999,
             group_key="qnngpu_tbt",
             latency_pattern=r"Test\s+QNN-GPU.*median=([\d.]+)\s*ms"),
        # M5: latency 측정 bin 부재 (qnn_oppkg_matmul_q40_correct 는 correctness only).
        # Phase C 진입 시 별 bin 작성 필요. 일단 optional.
        Cell("M5",  "q4_0", "opencl",
             "microbench_qnn_oppkg_matmul_q40_correct",
             args=[],
             env={"QNN_BACKEND_LIB": f"{DEVICE_QNN_LIBDIR}/libQnnGpu.so"},
             tolerance_max_abs=0.05, tolerance_cosine=0.999,
             optional=True),
        # M6 (FP32) 는 HTP 가 native 지원 안 함 — drop (paper conclusion 명료)
        # M6b/M7: qnn_executor_runner 의 output: "10 inference took X ms, avg Y ms"
        Cell("M6b", "f16",  "executorch",
             "qnn_executor_runner",
             args=["--model_path", "matmul_f16.pte",
                   "--input_list_path", "input_list.txt",
                   "--warm_up", "3", "--iteration", "10"],
             tolerance_max_abs=1e-2, tolerance_cosine=0.999,
             work_dir_override="/data/local/tmp/executorch",
             bin_dir_override="/data/local/tmp/executorch",
             latency_pattern=r"inference took [\d.]+\s*ms,\s*avg\s+([\d.]+)\s*ms"),
        Cell("M7",  "w8a8", "executorch",
             "qnn_executor_runner",
             args=["--model_path", "matmul_w8a8.pte",
                   "--input_list_path", "input_list.txt",
                   "--warm_up", "3", "--iteration", "10"],
             tolerance_max_abs=0.1, tolerance_cosine=0.99,
             work_dir_override="/data/local/tmp/executorch",
             bin_dir_override="/data/local/tmp/executorch",
             latency_pattern=r"inference took [\d.]+\s*ms,\s*avg\s+([\d.]+)\s*ms"),
        Cell("Ref", "f32",  "cpu-neon",
             "microbench_cpu_gemv_f32_ref",  # 미존재 — 후속 별 bin 작성 시 활성화
             args=[str(K), str(N), "6"],
             tolerance_max_abs=0.0, tolerance_cosine=1.0,
             optional=True),
    ]
    if enable:
        names = set(enable)
        for c in cells:
            c.enabled = c.name in names
    else:
        # default: non-optional 만. M1 (HTP) 은 #195 segfault risk 미확정이라
        # default 에서 빼고 explicit --cells 필요.
        default_off = {"M1", "Ref"}
        for c in cells:
            c.enabled = (not c.optional) and (c.name not in default_off)
    return cells


# ----------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------

@dataclass
class TrialResult:
    cell: str
    round_idx: int
    is_warmup: bool
    ok: bool
    latency_ms: Optional[float] = None
    max_abs_err: Optional[float] = None
    cosine: Optional[float] = None
    stderr: str = ""
    raw_stdout: str = ""
    thermal_before: Dict[str, float] = field(default_factory=dict)
    thermal_after: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


_LAT_PATTERNS = [
    # 표준 JSON
    ("json_latency_ms", lambda s: _parse_json_field(s, "latency_ms")),
    # "latency: 12.34 ms"
    ("latency_re", lambda s: _parse_regex(s, r"latency[:\s]+([\d.]+)\s*ms")),
    # "median: 12.34 ms"
    ("median_re", lambda s: _parse_regex(s, r"median[:\s]+([\d.]+)\s*ms")),
    # "tbt: 12.34 ms/tok"
    ("tbt_re", lambda s: _parse_regex(s, r"tbt[:\s]+([\d.]+)\s*ms")),
    # 일반 "<num> ms"
    ("any_ms_re", lambda s: _parse_regex(s, r"([\d.]+)\s*ms")),
]


def _parse_json_field(s: str, key: str) -> Optional[float]:
    for line in s.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
            if key in j:
                return float(j[key])
        except Exception:
            continue
    return None


def _parse_regex(s: str, pat: str) -> Optional[float]:
    import re
    m = re.search(pat, s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def parse_trial_output(
    stdout: str,
    custom_pattern: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (latency_ms, max_abs_err, cosine).

    custom_pattern 이 주어지면 그것만 사용 (multi-key bin 의 cell-specific 추출).
    """
    lat = None
    if custom_pattern:
        lat = _parse_regex(stdout, custom_pattern)
    if lat is None:
        for _, fn in _LAT_PATTERNS:
            v = fn(stdout)
            if v is not None:
                lat = v
                break
    err = _parse_json_field(stdout, "max_abs_err")
    if err is None:
        err = _parse_regex(stdout, r"max[_\s]abs[_\s]err[:\s]+([\d.eE+-]+)")
    cos = _parse_json_field(stdout, "cosine")
    if cos is None:
        cos = _parse_regex(stdout, r"cosine[:\s]+([\d.eE+-]+)")
    return lat, err, cos


# ----------------------------------------------------------------------
# Outlier rejection + stats
# ----------------------------------------------------------------------

def tukey_iqr(values: List[float], k: float = 1.5) -> Tuple[List[float], List[float]]:
    """Tukey 1.5×IQR rule. Returns (filtered, outliers)."""
    if len(values) < 4:
        return values[:], []
    q1, q3 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    filtered = [v for v in values if lo <= v <= hi]
    outliers = [v for v in values if v < lo or v > hi]
    return filtered, outliers


def cell_stats(trials: List[TrialResult]) -> Dict[str, object]:
    measure = [t for t in trials if (not t.is_warmup) and t.ok and t.latency_ms is not None]
    if not measure:
        return {"n": 0, "n_valid": 0, "median": None, "cv_pct": None}
    raw = [t.latency_ms for t in measure]
    filtered, outliers = tukey_iqr(raw)
    if not filtered:
        filtered = raw
    median = statistics.median(filtered)
    stdev = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
    cv_pct = (stdev / median * 100.0) if median > 0 else None
    quantiles = statistics.quantiles(filtered, n=4) if len(filtered) >= 4 else [None]*3
    max_abs = [t.max_abs_err for t in measure if t.max_abs_err is not None]
    cos = [t.cosine for t in measure if t.cosine is not None]
    return {
        "n_raw": len(raw),
        "n_outlier": len(outliers),
        "n_valid": len(filtered),
        "median_ms": round(median, 4),
        "stdev_ms": round(stdev, 4),
        "cv_pct": round(cv_pct, 2) if cv_pct is not None else None,
        "p25_ms": round(quantiles[0], 4) if quantiles[0] else None,
        "p75_ms": round(quantiles[2], 4) if quantiles[2] else None,
        "min_ms": round(min(filtered), 4),
        "max_ms": round(max(filtered), 4),
        "max_abs_err_max": round(max(max_abs), 6) if max_abs else None,
        "cosine_min": round(min(cos), 6) if cos else None,
    }


# ----------------------------------------------------------------------
# Cooldown / thermal control
# ----------------------------------------------------------------------

def wait_thermal_below(adb: Adb, target_c: float, max_wait: int, label: str, log_fn) -> bool:
    start = time.time()
    while True:
        name, t = adb.max_zone_temp()
        elapsed = time.time() - start
        log_fn(f"  [{label}] elapsed={elapsed:.0f}s max={name}={t:.1f}°C target<{target_c:.0f}°C")
        if t <= target_c:
            return True
        if elapsed >= max_wait:
            return False
        time.sleep(15)


def cooldown(adb: Adb, min_s: int, max_s: int, label: str, log_fn) -> None:
    log_fn(f"[cooldown:{label}] min={min_s}s")
    time.sleep(min_s)
    if not wait_thermal_below(adb, RECOVERY_TEMP_C + 4.0, max_s - min_s, label, log_fn):
        log_fn(f"  [cooldown:{label}] extra wait timeout — proceeding anyway")


# ----------------------------------------------------------------------
# Trial runner
# ----------------------------------------------------------------------

def run_trial(
    adb: Adb,
    cell: Cell,
    round_idx: int,
    is_warmup: bool,
    work_dir: str,
    log_fn,
) -> TrialResult:
    thermal_before = adb.read_zones()
    max_before = max(thermal_before.values())
    if max_before >= TRIGGER_TEMP_C:
        log_fn(f"  [trigger] max={max_before:.1f}°C ≥ {TRIGGER_TEMP_C}°C — extra cooldown")
        if not wait_thermal_below(adb, RECOVERY_TEMP_C, COOLDOWN_TRIGGER_S, "trigger-recovery", log_fn):
            return TrialResult(
                cell=cell.name, round_idx=round_idx, is_warmup=is_warmup, ok=False,
                error=f"thermal trigger not recovered (>{RECOVERY_TEMP_C}°C)",
                thermal_before=thermal_before,
            )
        thermal_before = adb.read_zones()

    cmd = cell.adb_cmd(work_dir)
    try:
        t0 = time.time()
        p = subprocess.run(
            ["adb", "-s", adb.serial, "shell", cmd],
            capture_output=True, text=True, timeout=180,
        )
        elapsed_wall = (time.time() - t0) * 1000.0
        ok = p.returncode == 0
        stdout = p.stdout
        stderr = p.stderr
    except subprocess.TimeoutExpired as e:
        return TrialResult(
            cell=cell.name, round_idx=round_idx, is_warmup=is_warmup, ok=False,
            error=f"timeout after 180s",
            thermal_before=thermal_before,
        )

    thermal_after = adb.read_zones()
    # Executorch (qnn_executor_runner) 는 logging 을 stderr 로 출력하므로
    # parse 시 stdout + stderr 양쪽 합쳐서 본다.
    combined = (stdout or "") + "\n" + (stderr or "")
    lat, err, cos = parse_trial_output(combined, cell.latency_pattern)
    parsed_latency = lat is not None
    if lat is None:
        # Crash (signal kill 등) 시에만 latency 없음. wall-clock 으로 폴백.
        lat = elapsed_wall

    accuracy_ok = True
    if err is not None and err > cell.tolerance_max_abs:
        accuracy_ok = False
    if cos is not None and cos < cell.tolerance_cosine:
        accuracy_ok = False

    # measurement validity 의 핵심 = latency 가 stdout 에서 parse 되었는가.
    # binary returncode != 0 자체는 본 매트릭스 입장에서 fatal 아님 (verdict
    # PASS/FAIL 의 표현일 수 있음 — 예: qnngpu_matmul_tbt 가 RED 시 exit 1).
    valid = parsed_latency and accuracy_ok

    return TrialResult(
        cell=cell.name, round_idx=round_idx, is_warmup=is_warmup,
        ok=valid,
        latency_ms=lat, max_abs_err=err, cosine=cos,
        stderr=stderr[-400:] if stderr else "",
        raw_stdout=stdout[-1200:] if stdout else "",
        thermal_before=thermal_before, thermal_after=thermal_after,
        error=None if valid else (
            "binary crashed (no latency in stdout)" if not parsed_latency
            else "accuracy out of tolerance"
        ),
    )


# ----------------------------------------------------------------------
# Matrix orchestration
# ----------------------------------------------------------------------

def run_matrix(
    adb: Adb,
    cells: List[Cell],
    out_dir: Path,
    rounds: int,
    seed: int,
    log_fn,
) -> Dict[str, List[TrialResult]]:
    enabled = [c for c in cells if c.enabled]
    log_fn(f"[matrix] {len(enabled)} cells × {rounds} rounds = {len(enabled)*rounds} trials")
    log_fn(f"[matrix] cells: {[c.name for c in enabled]}")

    work_dir = DEVICE_WORK_DIR
    adb.shell(f"mkdir -p {work_dir}", check=False)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    thermal_log_path = out_dir / "thermal_log.csv"
    seq_path = out_dir / "round_sequence.json"

    with thermal_log_path.open("w") as tf:
        tf.write("timestamp,phase,cell,round_idx," + ",".join(ZONES.keys()) + "\n")

    rng = random.Random(seed)
    sequence: List[List[str]] = []
    results: Dict[str, List[TrialResult]] = {c.name: [] for c in enabled}

    log_fn(f"[matrix] session warmup {SESSION_WARMUP_S}s")
    time.sleep(SESSION_WARMUP_S)

    for r in range(rounds):
        order = [c.name for c in enabled]
        rng.shuffle(order)
        sequence.append(order)
        is_warmup = (r < WARMUP_TRIALS)
        log_fn(f"\n=== round {r+1}/{rounds} ({'WARMUP' if is_warmup else 'MEASURE'}) order={order} ===")

        for cname in order:
            cell = next(c for c in enabled if c.name == cname)

            # Inter-cell cooldown (첫 cell 제외)
            if results[cname]:
                cooldown(adb, COOLDOWN_MIN_S, COOLDOWN_MAX_S, f"inter-cell {cname}", log_fn)

            log_fn(f"\n--- {cname} round {r+1} ({'warmup' if is_warmup else 'measure'}) ---")
            tr = run_trial(adb, cell, r, is_warmup, work_dir, log_fn)
            results[cname].append(tr)

            # Per-trial json
            tr_dict = asdict(tr)
            with (raw_dir / f"{cname}_round{r:02d}.json").open("w") as f:
                json.dump(tr_dict, f, indent=2)

            # Thermal log
            with thermal_log_path.open("a") as tf:
                ts = int(time.time())
                bb = ",".join(f"{tr.thermal_before.get(k, 0):.2f}" for k in ZONES.keys())
                aa = ",".join(f"{tr.thermal_after.get(k, 0):.2f}" for k in ZONES.keys())
                tf.write(f"{ts},before,{cname},{r},{bb}\n")
                tf.write(f"{ts},after,{cname},{r},{aa}\n")

            log_fn(
                f"  result: ok={tr.ok} latency={tr.latency_ms:.3f}ms "
                f"err={tr.max_abs_err} cos={tr.cosine} "
                f"max_before={max(tr.thermal_before.values()):.1f}°C "
                f"max_after={max(tr.thermal_after.values()):.1f}°C"
            )

            time.sleep(INTER_TRIAL_S)

        # Inter-round cooldown (마지막 round 제외)
        if r < rounds - 1:
            cooldown(adb, INTER_ROUND_S, INTER_ROUND_MAX_S, f"inter-round {r+1}->{r+2}", log_fn)

    with seq_path.open("w") as f:
        json.dump(sequence, f, indent=2)

    return results


# ----------------------------------------------------------------------
# Aggregation + report
# ----------------------------------------------------------------------

def aggregate(results: Dict[str, List[TrialResult]], cells: List[Cell], out_dir: Path) -> None:
    csv_path = out_dir / "aggregated.csv"
    summary_path = out_dir / "summary.json"

    summary = {}
    with csv_path.open("w") as f:
        f.write("cell,dtype,backend,n_raw,n_outlier,n_valid,median_ms,stdev_ms,cv_pct,"
                "p25_ms,p75_ms,min_ms,max_ms,max_abs_err_max,cosine_min\n")
        for c in cells:
            if not c.enabled:
                continue
            st = cell_stats(results.get(c.name, []))
            summary[c.name] = {"meta": {"dtype": c.dtype, "backend": c.backend}, **st}
            f.write(",".join(str(x) for x in [
                c.name, c.dtype, c.backend,
                st.get("n_raw", 0), st.get("n_outlier", 0), st.get("n_valid", 0),
                st.get("median_ms"), st.get("stdev_ms"), st.get("cv_pct"),
                st.get("p25_ms"), st.get("p75_ms"),
                st.get("min_ms"), st.get("max_ms"),
                st.get("max_abs_err_max"), st.get("cosine_min"),
            ]) + "\n")

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


def generate_report(results: Dict[str, List[TrialResult]], cells: List[Cell], out_dir: Path) -> None:
    md = out_dir / "report.md"
    lines = [
        f"# QNN microbench matrix 결과",
        f"",
        f"- 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Shape: K={K}, N={N} (Qwen 2.5-1.5b FFN gate)",
        f"- Trial protocol: warmup {WARMUP_TRIALS} + measure {MEASURE_TRIALS}",
        f"",
        "## 매트릭스 결과",
        "",
        "| Cell | dtype | backend | n_valid | n_outlier | median (ms) | CV (%) | max_abs_err | cosine | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        if not c.enabled:
            continue
        st = cell_stats(results.get(c.name, []))
        cv = st.get("cv_pct")
        cv_status = "GREEN" if cv is not None and cv < 5 else ("YELLOW" if cv is not None and cv < 10 else "RED")
        if st["n_valid"] == 0:
            cv_status = "FAILED"
        lines.append(f"| {c.name} | {c.dtype} | {c.backend} | "
                     f"{st['n_valid']}/{st.get('n_raw',0)} | {st.get('n_outlier','-')} | "
                     f"{st.get('median_ms','-')} | {cv if cv is not None else '-'} | "
                     f"{st.get('max_abs_err_max','-')} | {st.get('cosine_min','-')} | {cv_status} |")
    lines.append("")
    lines.append("## Fair-pair 분석")
    lines.append("")
    pairs = [
        ("FP32", ["M1", "M6"]),
        ("F16", ["M1b", "M3", "M4", "M6b"]),
        ("W8A8", ["M2", "M7"]),
        ("Production ref", ["M5"]),
    ]
    for row_name, cell_names in pairs:
        lines.append(f"### {row_name} row")
        lines.append("")
        for cn in cell_names:
            st = cell_stats(results.get(cn, []))
            if st.get("median_ms") is not None:
                lines.append(f"- **{cn}**: {st['median_ms']} ms (CV {st['cv_pct']}%)")
            else:
                lines.append(f"- **{cn}**: SKIPPED or FAILED")
        lines.append("")
    md.write_text("\n".join(lines))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="QNN microbench matrix driver")
    parser.add_argument("--serial", default="R3CY408S5SB",
                        help="ADB device serial (default: R3CY408S5SB = S25 measurement target)")
    parser.add_argument("--cells", default="",
                        help="Comma-separated cell names. Empty = all non-optional + Ref.")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help=f"Total rounds (default {ROUNDS} = {WARMUP_TRIALS} warmup + {MEASURE_TRIALS} measure)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--dry-run-setup", action="store_true",
                        help="Skip measurement; just verify ADB + zones + zombie check")
    parser.add_argument("--quick", action="store_true",
                        help="단축 cooldown (inter-cell 10s, inter-round 30s) — protocol 검증용")
    args = parser.parse_args()

    if args.quick:
        global COOLDOWN_MIN_S, COOLDOWN_MAX_S, INTER_ROUND_S, INTER_ROUND_MAX_S, SESSION_WARMUP_S
        COOLDOWN_MIN_S = 10
        COOLDOWN_MAX_S = 60
        INTER_ROUND_S = 30
        INTER_ROUND_MAX_S = 90
        SESSION_WARMUP_S = 15

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "driver.log"
    log_f = log_path.open("w")
    def log_fn(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    enable = [s.strip() for s in args.cells.split(",") if s.strip()]
    cells = build_cells(enable)

    log_fn(f"[init] device={args.serial}")
    log_fn(f"[init] out={out_dir}")
    log_fn(f"[init] enabled cells: {[c.name for c in cells if c.enabled]}")

    adb = Adb(args.serial)

    # Preflight: zombie
    zombies = adb.zombie_check()
    if zombies:
        log_fn(f"[preflight] FAIL — zombies on device:")
        for z in zombies:
            log_fn(f"  {z}")
        log_fn(f"[preflight] kill them first or use a different device")
        return 3
    log_fn(f"[preflight] zombie check OK")

    # Preflight: thermal start
    name, t = adb.max_zone_temp()
    log_fn(f"[preflight] max_zone={name}={t:.1f}°C (start<{START_TEMP_C}°C)")
    if t > START_TEMP_C:
        log_fn(f"[preflight] device too hot — cooling down")
        if not wait_thermal_below(adb, START_TEMP_C, 300, "preflight", log_fn):
            log_fn(f"[preflight] cooldown timeout — abort")
            return 4

    # Preflight: airplane mode + screen state (warn only)
    airplane = adb.shell("settings get global airplane_mode_on", check=False)
    log_fn(f"[preflight] airplane_mode_on={airplane!r}")
    if airplane.strip() != "1":
        log_fn(f"[preflight] WARN: airplane mode is OFF — measurement noise risk")

    if args.dry_run_setup:
        log_fn(f"[dry-run-setup] preflight OK, exiting before measurement")
        log_f.close()
        return 0

    # Env snapshot
    env = {
        "device_serial": args.serial,
        "model": adb.shell("getprop ro.product.model", check=False),
        "soc": adb.shell("getprop ro.soc.model", check=False),
        "build_id": adb.shell("getprop ro.build.id", check=False),
        "kernel": adb.shell("uname -r", check=False),
        "rounds": args.rounds,
        "seed": args.seed,
        "warmup_trials": WARMUP_TRIALS,
        "measure_trials": MEASURE_TRIALS,
        "zones": ZONES,
        "k": K,
        "n": N,
        "trigger_temp_c": TRIGGER_TEMP_C,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))

    # Run!
    try:
        results = run_matrix(adb, cells, out_dir, args.rounds, args.seed, log_fn)
    except KeyboardInterrupt:
        log_fn(f"[interrupted]")
        log_f.close()
        return 130
    except Exception as e:
        log_fn(f"[error] {type(e).__name__}: {e}")
        log_f.close()
        raise

    aggregate(results, cells, out_dir)
    generate_report(results, cells, out_dir)

    log_fn(f"\n[done] outputs:")
    log_fn(f"  - {out_dir}/raw/")
    log_fn(f"  - {out_dir}/aggregated.csv")
    log_fn(f"  - {out_dir}/summary.json")
    log_fn(f"  - {out_dir}/thermal_log.csv")
    log_fn(f"  - {out_dir}/round_sequence.json")
    log_fn(f"  - {out_dir}/report.md")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
