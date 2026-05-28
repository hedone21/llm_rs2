#!/usr/bin/env python3
"""llamacpp_backend_ops_driver.py — L.cpp test-backend-ops same-device driver.

P2 phase: lcpp.{cpu, gpu, htp} 3 sub-backend × 12 op × 2 dtype = 측정 driver.

핵심 protocol (plan_qnn_microbench_measurement_protocol_2026_05_26.md inherit):
- warmup 3 + measure 10 trial per cell
- round-robin shuffle (cell 순서 발열 누적 편향 제거)
- 8 thermal zone (CPU little/mid/prime + GPU + Hexagon HVX/HMX + DDR) polling
- 50°C trigger / 60s session warmup / 45~180s inter-cell cooldown
- Tukey 1.5×IQR outlier rejection
- taskset 3f (6T pinning, CLAUDE.md 권장)
- raw + aggregated.csv + thermal_log.csv + report.md 출력

binary: /data/local/tmp/test-backend-ops
  - LD_PRELOAD: libggml-hexagon-stub.so (CPU/GPU 단독 실행 시 HTP0 crash 방지)
  - -b GPUOpenCL: Adreno 830 GPU backend
  - -b CPU: ARM64 NEON CPU backend
  - HTP0: session create 실패 → ⚠ device_init_failed 마킹

lcpp.htp 처리 정책 (matrix.md v2 §v2-§5.5 및 사용자 결정 2026-05-26):
  - HTP0 device session 생성 자체 실패 (FastRPC capability query failed, unsigned PD failed)
  - 모든 perf cell: ⚠ device_init_failed (latency=null)
  - support 정보만 reference (test-backend-ops test 모드로 HTP0 직접 실행 시도)
  - 기존 v1 에서의 "dspqueue_write failed: 0x0000000c" 패턴도 ⚠ perf_abort 로 처리

perf case 유무 (실측 확인):
  있음: ADD, SOFT_MAX, ROPE, MUL_MAT, CPY
  없음: SILU (UNARY), MUL, RMS_NORM, GET_ROWS, SCALE, SET_ROWS, FLASH_ATTN_EXT
  → 없는 op 는 latency_us=null, status=no_perf_case (support 는 test 모드로 확인)

S25 zone naming (Phase A.2 결과):
- nsphvx-* (zone 39~41): Hexagon HVX
- nsphmx-* (zone 42~45): Hexagon HMX
- gpuss-* (zone 23~30): Adreno
- cpu-0-*/cpu-1-*: CPU clusters
- ddr (zone 46)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVICE_BIN_DIR = "/data/local/tmp"
DEVICE_WORK_DIR = "/data/local/tmp"

LCPP_BINARY = "test-backend-ops"
LCPP_STUB_SO = "libggml-hexagon-stub.so"  # HTP crash 방지 stub

# Host binary path (존재 여부 확인용)
HOST_LCPP_DIR = Path("/home/go/Workspace/llama.cpp/build-snapdragon/bin")
HOST_STUB_SO = Path("/tmp/libggml-hexagon-stub.so")

# ---------------------------------------------------------------------------
# Thermal monitoring (microbench_qnn_matrix.py 동일)
# ---------------------------------------------------------------------------

ZONES: Dict[str, int] = {
    "cpu_little": 1,
    "cpu_mid":    9,
    "cpu_prime":  16,
    "gpu_5":      28,
    "gpu_7":      30,
    "hex_vec":    39,
    "hex_mat":    42,
    "ddr":        46,
}

TRIGGER_TEMP_C = 50.0
RECOVERY_TEMP_C = 38.0
DANGER_TEMP_C = 60.0
START_TEMP_C = 40.0

SESSION_WARMUP_S = 60
COOLDOWN_MIN_S = 45
COOLDOWN_MAX_S = 180
COOLDOWN_TRIGGER_S = 600
INTER_TRIAL_S = 5
INTER_ROUND_S = 300
INTER_ROUND_MAX_S = 600

WARMUP_TRIALS = 3
MEASURE_TRIALS = 10
ROUNDS = WARMUP_TRIALS + MEASURE_TRIALS  # 13

# ---------------------------------------------------------------------------
# Qwen 2.5-1.5B shape 정의 (matrix.md v2 §v2-§3)
# ---------------------------------------------------------------------------

QWEN_SHAPES = {
    # MUL_MAT 3 shape
    # NOTE: test-backend-ops 의 MUL_MAT perf 는 고정 shape 세트 (llama 7B 기반).
    # Qwen 2.5-1.5B exact shape (m=8960/151936/2048, k=1536) 는 없음.
    # → params_filter=None: dtype 만 필터링, 기본 GEMV shape (m=4096,k=14336) 사용.
    # paper note: "L.cpp perf uses fixed GEMV shape m=4096,k=14336; Qwen shapes unavailable"
    "mm_ffn": {
        "op": "MUL_MAT",
        "desc": "FFN GEMV decode (test-backend-ops 기본: m=4096,k=14336; Qwen m=8960,k=1536 없음)",
        "params_filter": r"n=1,k=14336",  # GEMV decode (n=1) shape 우선
    },
    "mm_lmh": {
        "op": "MUL_MAT",
        "desc": "LM head GEMV (test-backend-ops 기본 shape 사용; Qwen m=151936 없음)",
        "params_filter": r"n=1,k=14336",
    },
    "mm_qkv": {
        "op": "MUL_MAT",
        "desc": "QKV GEMV (test-backend-ops 기본 shape 사용; Qwen fused N=2048 없음)",
        "params_filter": r"n=1,k=14336",
    },
    # Single-shape ops
    "RMS_NORM": {
        "op": "RMS_NORM",
        "desc": "[1, 1536] eps=1e-6",
        "params_filter": None,
    },
    "ROPE": {
        "op": "ROPE",
        "desc": "head_dim=128 mode=8 (Qwen2) n=12",
        "params_filter": r"ne_a=\[128,12",
    },
    "FLASH_ATTN_EXT": {
        "op": "FLASH_ATTN",
        "desc": "hs=128 nh=12 nkv=2 ctx=1024",
        "params_filter": None,
    },
    "GET_ROWS": {
        "op": "GET_ROWS",
        "desc": "vocab=151936 embed",
        "params_filter": None,
    },
    "SILU": {
        "op": "SILU",
        "desc": "[1, 8960]",
        "params_filter": None,
    },
    "MUL": {
        "op": "MUL",
        "desc": "[1, 8960] gate·up",
        "params_filter": None,
    },
    "ADD": {
        "op": "ADD",
        "desc": "[4096] (test-backend-ops 고정 shape, Qwen [1536] 미정의)",
        "params_filter": None,  # test-backend-ops ADD 는 ne=[4096] 만 존재
    },
    "SOFT_MAX": {
        "op": "SOFT_MAX",
        "desc": "[1024×N] 대표 shape",
        "params_filter": r"ne=\[1024",
    },
    "SCALE": {
        "op": "SCALE",
        "desc": "1/sqrt(d_k)",
        "params_filter": None,
    },
    "CPY": {
        "op": "CPY",
        "desc": "F32->F16 [2,1,128] KV store",
        "params_filter": r"ne=\[128",
    },
    "SET_ROWS": {
        "op": "SET_ROWS",
        "desc": "KV scatter [2,1,128]->cache",
        "params_filter": None,
    },
}

# perf test case 있는 op (실측 확인 2026-05-28)
OPS_WITH_PERF_CASE = {"MUL_MAT", "ADD", "SOFT_MAX", "ROPE", "CPY"}

# Dtype → test-backend-ops 파라미터 필터 패턴
DTYPE_PATTERNS = {
    "f16":   r"type[_a]?=f16",
    "q4_0":  r"type[_a]?=q4_0",
}

# ---------------------------------------------------------------------------
# Backend 정의
# ---------------------------------------------------------------------------

BACKENDS = {
    "lcpp.cpu": {
        "flag": "CPU",
        "use_stub": True,   # LD_PRELOAD stub 필요 (HTP0 crash 방지)
        "desc": "llama.cpp ARM64 NEON CPU",
    },
    "lcpp.gpu": {
        "flag": "GPUOpenCL",
        "use_stub": False,  # GPU 단독 실행 시 crash 없음 (CPU skip 후 HTP nullptr → abort 발생)
        "use_stub_for_crash": True,  # 실제로는 stub 필요 (3-device loop crash)
        "desc": "llama.cpp OpenCL Adreno 830",
    },
    "lcpp.htp": {
        "flag": "HTP0",
        "use_stub": False,
        "htp_device_fail": True,   # device session 생성 자체 실패
        "desc": "llama.cpp ggml-hexagon HTP0 (device_init_failed)",
    },
}

# ---------------------------------------------------------------------------
# Cell 정의
# ---------------------------------------------------------------------------

@dataclass
class LcppCell:
    """test-backend-ops 측정 단위 (1 op × 1 backend × 1 dtype × 1 shape)."""
    cell_id: str           # e.g. lcpp_cpu_MUL_MAT_f16_mm_ffn
    backend_key: str       # lcpp.{cpu,gpu,htp}
    backend_flag: str      # CPU, GPUOpenCL, HTP0
    op_name: str           # test-backend-ops -o OP
    shape_key: str         # shape label
    dtype: str             # f16 or q4_0
    desc: str
    params_filter: Optional[str]  # -p regex
    has_perf_case: bool    # test-backend-ops perf 케이스 존재 여부
    use_stub: bool         # LD_PRELOAD stub 사용 여부
    htp_device_fail: bool = False  # HTP0 device init 실패
    enabled: bool = True
    # 마킹 상태
    cell_status: str = "+"  # +/✓/✗/⚠/—

    def is_measurable(self) -> bool:
        """실제 latency 측정 가능 여부."""
        if self.htp_device_fail:
            return False
        if not self.has_perf_case:
            return False
        return True


def build_cells(
    ops_filter: Optional[List[str]] = None,
    backends_filter: Optional[List[str]] = None,
    dtypes_filter: Optional[List[str]] = None,
) -> List[LcppCell]:
    """측정 대상 cell 목록 생성."""
    cells = []

    shapes_to_measure = list(QWEN_SHAPES.items())

    dtypes = dtypes_filter or ["f16", "q4_0"]
    backends = backends_filter or list(BACKENDS.keys())

    for shape_key, shape_def in shapes_to_measure:
        op_name = shape_def["op"]

        # op 필터
        if ops_filter and op_name not in ops_filter:
            # shape_key 도 확인 (MUL_MAT의 경우 shape_key가 mm_ffn 등)
            if shape_key not in ops_filter:
                continue

        for dtype in dtypes:
            # activation-only op의 Q4_0 row = — 마킹 (fair 비교 부적합)
            activation_only_ops = {
                "RMS_NORM", "ROPE", "FLASH_ATTN_EXT", "SILU", "MUL", "ADD",
                "SOFT_MAX", "SCALE", "CPY", "SET_ROWS",
            }
            if dtype == "q4_0" and op_name in activation_only_ops:
                # — 마킹 cell: 스킵 (fair 부적합)
                continue

            for bk in backends:
                bdef = BACKENDS[bk]
                has_perf = op_name in OPS_WITH_PERF_CASE

                # HTP device fail 여부
                htp_fail = bdef.get("htp_device_fail", False)

                # cell_id: lcpp_{cpu/gpu/htp}_{OP}_{dtype}[_{shape_id}]
                bk_short = bk.replace("lcpp.", "")
                if shape_key.startswith("mm_"):
                    cid = f"lcpp_{bk_short}_{op_name}_{dtype}_{shape_key}"
                else:
                    cid = f"lcpp_{bk_short}_{op_name}_{dtype}"

                # 마킹 상태
                if htp_fail:
                    status = "⚠"
                elif not has_perf:
                    status = "⚠"  # perf case 없음
                else:
                    status = "+"

                # stub 사용: CPU는 필수, GPU도 실제로 필요 (3-device loop crash)
                use_stub = bdef.get("use_stub", False) or bdef.get("use_stub_for_crash", False)

                cell = LcppCell(
                    cell_id=cid,
                    backend_key=bk,
                    backend_flag=bdef["flag"],
                    op_name=op_name,
                    shape_key=shape_key,
                    dtype=dtype,
                    desc=shape_def["desc"],
                    params_filter=shape_def.get("params_filter"),
                    has_perf_case=has_perf,
                    use_stub=use_stub,
                    htp_device_fail=htp_fail,
                    cell_status=status,
                )
                cells.append(cell)

    return cells


# ---------------------------------------------------------------------------
# ADB wrapper (microbench_qnn_matrix.py 동일)
# ---------------------------------------------------------------------------

class Adb:
    def __init__(self, serial: str):
        self.serial = serial

    def shell(self, cmd: str, timeout: int = 300, check: bool = True) -> str:
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
        paths = " ".join(
            f"/sys/class/thermal/thermal_zone{z}/temp" for z in ZONES.values()
        )
        raw = self.shell(f"cat {paths}", timeout=10)
        temps = [
            int(x.strip()) for x in raw.split() if x.strip().lstrip("-").isdigit()
        ]
        if len(temps) != len(ZONES):
            raise RuntimeError(f"zone count mismatch: {len(temps)} vs {len(ZONES)}")
        return {name: temps[i] / 1000.0 for i, name in enumerate(ZONES.keys())}

    def max_zone_temp(self) -> Tuple[str, float]:
        temps = self.read_zones()
        name = max(temps, key=temps.get)
        return name, temps[name]

    def zombie_check(self) -> List[str]:
        out = self.shell(
            "ps -A 2>/dev/null | grep -E '(test-backend-ops|microbench|generate)' | grep -v grep",
            check=False,
        )
        return [l.strip() for l in out.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Perf output parsing
# ---------------------------------------------------------------------------

# 출력 형식:
#   OP_NAME(params...):  <N> runs - <latency> us/run - ...
PERF_LINE_RE = re.compile(
    r"^\s+\S.*?:\s+(\d+)\s+runs\s+-\s+([\d.]+)\s+us/run",
    re.MULTILINE,
)

# HTP abort 패턴들
HTP_ABORT_PATTERNS = [
    "dspqueue_write failed",
    "failed to create device/session",
    "FastRPC capability query failed",
    "failed to query HTP version",
]

# "not supported" 패턴
NOT_SUPPORTED_RE = re.compile(r"not supported")


def build_adb_cmd(
    cell: LcppCell,
    work_dir: str,
    mode: str = "perf",
) -> str:
    """test-backend-ops ADB 실행 명령 생성."""
    ld_path = f"{work_dir}:/vendor/lib64:/system/lib64"
    env_parts = [f"LD_LIBRARY_PATH={ld_path}"]

    if cell.use_stub:
        env_parts.append(f"LD_PRELOAD={work_dir}/{LCPP_STUB_SO}")

    env_str = " ".join(env_parts)

    cmd_parts = [
        f"cd {work_dir}",
        f"{env_str}",
        f"taskset 3f ./{LCPP_BINARY} {mode}",
        f"-o {cell.op_name}",
        f"-b {cell.backend_flag}",
    ]

    if cell.params_filter:
        cmd_parts.append(f"-p '{cell.params_filter}'")

    # dtype 필터: MUL_MAT 의 경우 type_a 로 필터
    if cell.dtype == "f16":
        dtype_flag = "f16"
    else:
        dtype_flag = "q4_0"
    # -p 를 이미 지정한 경우에는 dtype 필터 추가 어려움 → 파싱에서 필터링
    # (test-backend-ops 의 -p 는 전체 param string regex)

    return " && ".join(cmd_parts[:1] + [" ".join(cmd_parts[1:])])


def parse_perf_output(
    stdout: str,
    stderr: str,
    cell: LcppCell,
) -> Tuple[Optional[float], str, str]:
    """stdout/stderr 파싱 → (latency_us, status, detail).

    Returns:
        latency_us: 해당 dtype의 latency (μs), None이면 측정 실패
        status: ok | dtype_f32_fallback | no_perf_case | not_supported |
                perf_abort | device_init_failed | parse_fail
        detail: 원인 문자열
    """
    combined = (stdout or "") + "\n" + (stderr or "")

    # HTP abort 패턴 검출
    for pat in HTP_ABORT_PATTERNS:
        if pat in combined:
            return None, "perf_abort", f"pattern={pat!r}"

    # device init 실패
    if "failed to create device/session" in combined:
        return None, "device_init_failed", "HTP0 session create failed"

    # perf 케이스 없음 (Backend OK but no perf lines)
    if not PERF_LINE_RE.search(stdout or ""):
        return None, "no_perf_case", "no perf test cases defined for this op"

    dtype_target = cell.dtype  # "f16" or "q4_0"

    def _extract_best(lines: List[str], dtype_filter: Optional[str]) -> Optional[float]:
        """dtype_filter 에 매칭되는 라인 중 best latency 추출.
        dtype_filter=None 이면 모든 라인 대상.
        """
        best = None
        for line in lines:
            if "not supported" in line:
                continue
            if dtype_filter and dtype_filter not in line:
                continue
            m = PERF_LINE_RE.match(line)
            if not m:
                continue
            lat = float(m.group(2))
            if cell.params_filter:
                if not re.search(cell.params_filter, line, re.IGNORECASE):
                    continue
            if best is None or lat < best:
                best = lat
        return best

    lines = (stdout or "").splitlines()

    # 1차: 정확한 dtype 매칭
    best_lat = _extract_best(lines, dtype_target)
    if best_lat is not None:
        return best_lat, "ok", ""

    # 2차 fallback: F16 요청 시 F32 케이스 사용
    # (ADD/ROPE/SOFT_MAX/CPY 등 test-backend-ops 가 F32 케이스만 제공)
    if dtype_target == "f16":
        best_f32 = _extract_best(lines, "f32")
        if best_f32 is not None:
            return (
                best_f32,
                "dtype_f32_fallback",
                "no f16 perf case; using f32 result (test-backend-ops perf 는 F32 케이스만 제공)",
            )

    # not supported 체크
    if "not supported" in (stdout or ""):
        return None, "not_supported", f"op not supported for dtype={dtype_target}"

    return None, "parse_fail", f"no matching {dtype_target} perf line found"


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    cell_id: str
    round_idx: int
    is_warmup: bool
    ok: bool
    latency_us: Optional[float] = None
    cell_status: str = "+"
    measurement_status: str = "ok"
    detail: str = ""
    raw_stdout: str = ""
    raw_stderr: str = ""
    thermal_before: Dict[str, float] = field(default_factory=dict)
    thermal_after: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


def run_trial(
    adb: Adb,
    cell: LcppCell,
    round_idx: int,
    is_warmup: bool,
    work_dir: str,
    log_fn,
) -> TrialResult:
    thermal_before = adb.read_zones()
    max_before = max(thermal_before.values())

    if max_before >= TRIGGER_TEMP_C:
        log_fn(f"  [trigger] max={max_before:.1f}°C ≥ {TRIGGER_TEMP_C}°C — extra cooldown")
        if not wait_thermal_below(
            adb, RECOVERY_TEMP_C, COOLDOWN_TRIGGER_S, "trigger-recovery", log_fn
        ):
            return TrialResult(
                cell_id=cell.cell_id,
                round_idx=round_idx,
                is_warmup=is_warmup,
                ok=False,
                error=f"thermal trigger not recovered (>{RECOVERY_TEMP_C}°C)",
                thermal_before=thermal_before,
            )
        thermal_before = adb.read_zones()

    # HTP device fail → 즉시 ⚠ 반환
    if cell.htp_device_fail:
        return TrialResult(
            cell_id=cell.cell_id,
            round_idx=round_idx,
            is_warmup=is_warmup,
            ok=True,  # "측정됨" (결과: device_init_failed)
            latency_us=None,
            cell_status="⚠",
            measurement_status="device_init_failed",
            detail="HTP0 session create failed on S25 SM8750 (unsigned PD error)",
            thermal_before=thermal_before,
            thermal_after=thermal_before,
        )

    # perf 케이스 없는 op → test 모드로 support 확인
    if not cell.has_perf_case:
        mode = "test"
    else:
        mode = "perf"

    cmd = build_adb_cmd(cell, work_dir, mode=mode)

    try:
        t0 = time.time()
        p = subprocess.run(
            ["adb", "-s", adb.serial, "shell", cmd],
            capture_output=True, text=True, timeout=600,
        )
        elapsed_wall_us = (time.time() - t0) * 1_000_000
        stdout = p.stdout or ""
        stderr = p.stderr or ""
    except subprocess.TimeoutExpired:
        return TrialResult(
            cell_id=cell.cell_id,
            round_idx=round_idx,
            is_warmup=is_warmup,
            ok=False,
            error="timeout after 600s",
            thermal_before=thermal_before,
        )

    thermal_after = adb.read_zones()

    if not cell.has_perf_case:
        # test 모드: support 확인만. latency 없음
        # 크래시 여부 확인
        if "GGML_ASSERT" in stderr or "Aborted" in stderr or "Segfault" in stderr:
            mstatus = "crash"
        elif "SUPPORTED" in stdout or "Backend" in stdout:
            mstatus = "no_perf_case"
        else:
            mstatus = "no_perf_case"
        return TrialResult(
            cell_id=cell.cell_id,
            round_idx=round_idx,
            is_warmup=is_warmup,
            ok=True,
            latency_us=None,
            cell_status="⚠",
            measurement_status=mstatus,
            detail="no perf test cases for this op in test-backend-ops",
            raw_stdout=stdout[-800:],
            raw_stderr=stderr[-400:],
            thermal_before=thermal_before,
            thermal_after=thermal_after,
        )

    # perf 모드: 파싱
    latency_us, mstatus, detail = parse_perf_output(stdout, stderr, cell)

    ok = latency_us is not None
    cstatus = "⚠" if not ok else "✓"

    return TrialResult(
        cell_id=cell.cell_id,
        round_idx=round_idx,
        is_warmup=is_warmup,
        ok=ok,
        latency_us=latency_us,
        cell_status=cstatus,
        measurement_status=mstatus,
        detail=detail,
        raw_stdout=stdout[-1200:],
        raw_stderr=stderr[-400:],
        thermal_before=thermal_before,
        thermal_after=thermal_after,
        error=None if ok else f"latency parse failed: {mstatus} — {detail}",
    )


# ---------------------------------------------------------------------------
# Thermal helpers (microbench_qnn_matrix.py 동일)
# ---------------------------------------------------------------------------

def wait_thermal_below(
    adb: Adb, target_c: float, max_wait: int, label: str, log_fn
) -> bool:
    start = time.time()
    while True:
        name, t = adb.max_zone_temp()
        elapsed = time.time() - start
        log_fn(
            f"  [{label}] elapsed={elapsed:.0f}s max={name}={t:.1f}°C target<{target_c:.0f}°C"
        )
        if t <= target_c:
            return True
        if elapsed >= max_wait:
            return False
        time.sleep(15)


def cooldown(adb: Adb, min_s: int, max_s: int, label: str, log_fn) -> None:
    log_fn(f"[cooldown:{label}] min={min_s}s")
    time.sleep(min_s)
    if not wait_thermal_below(
        adb, RECOVERY_TEMP_C + 4.0, max_s - min_s, label, log_fn
    ):
        log_fn(f"  [cooldown:{label}] extra wait timeout — proceeding anyway")


# ---------------------------------------------------------------------------
# Tukey IQR
# ---------------------------------------------------------------------------

def tukey_iqr(
    values: List[float], k: float = 1.5
) -> Tuple[List[float], List[float]]:
    if len(values) < 4:
        return values[:], []
    q1 = statistics.quantiles(values, n=4)[0]
    q3 = statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    filtered = [v for v in values if lo <= v <= hi]
    outliers = [v for v in values if v < lo or v > hi]
    return filtered, outliers


def cell_stats(trials: List[TrialResult]) -> Dict:
    measure = [
        t for t in trials if not t.is_warmup and t.ok and t.latency_us is not None
    ]
    if not measure:
        # latency 없는 경우 (no_perf_case / device_init_failed)
        all_statuses = list({t.measurement_status for t in trials if not t.is_warmup})
        mstatus = all_statuses[0] if len(all_statuses) == 1 else (
            all_statuses[0] if all_statuses else "no_measure_trials"
        )
        return {
            "n": 0,
            "n_valid": 0,
            "median_us": None,
            "cv_pct": None,
            "measurement_status": mstatus,
        }
    raw = [t.latency_us for t in measure]
    filtered, outliers = tukey_iqr(raw)
    if not filtered:
        filtered = raw
    median = statistics.median(filtered)
    stdev = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
    cv_pct = (stdev / median * 100.0) if median > 0 else None
    quantiles = (
        statistics.quantiles(filtered, n=4) if len(filtered) >= 4 else [None] * 3
    )
    return {
        "n_raw": len(raw),
        "n_outlier": len(outliers),
        "n_valid": len(filtered),
        "median_us": round(median, 3),
        "median_ms": round(median / 1000.0, 4),
        "stdev_us": round(stdev, 3),
        "cv_pct": round(cv_pct, 2) if cv_pct is not None else None,
        "p25_us": round(quantiles[0], 3) if quantiles[0] is not None else None,
        "p75_us": round(quantiles[2], 3) if quantiles[2] is not None else None,
        "min_us": round(min(filtered), 3),
        "max_us": round(max(filtered), 3),
        "measurement_status": "ok",
    }


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

def run_matrix(
    adb: Adb,
    cells: List[LcppCell],
    out_dir: Path,
    rounds: int,
    seed: int,
    log_fn,
    dry_run: bool = False,
) -> Dict[str, List[TrialResult]]:
    enabled = [c for c in cells if c.enabled]
    log_fn(
        f"[matrix] {len(enabled)} cells × {rounds} rounds = {len(enabled) * rounds} trials"
    )
    log_fn(f"[matrix] cells: {[c.cell_id for c in enabled]}")

    work_dir = DEVICE_WORK_DIR
    adb.shell(f"mkdir -p {work_dir}", check=False)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    thermal_log_path = out_dir / "thermal_log.csv"
    seq_path = out_dir / "round_sequence.json"

    with thermal_log_path.open("w") as tf:
        tf.write(
            "timestamp,phase,cell_id,round_idx," + ",".join(ZONES.keys()) + "\n"
        )

    rng = random.Random(seed)
    sequence: List[List[str]] = []
    results: Dict[str, List[TrialResult]] = {c.cell_id: [] for c in enabled}

    log_fn(f"[matrix] session warmup {SESSION_WARMUP_S}s")
    if not dry_run:
        time.sleep(SESSION_WARMUP_S)

    for r in range(rounds):
        order = [c.cell_id for c in enabled]
        rng.shuffle(order)
        sequence.append(order)
        is_warmup = r < WARMUP_TRIALS
        log_fn(
            f"\n=== round {r+1}/{rounds} ({'WARMUP' if is_warmup else 'MEASURE'}) ==="
        )

        for cid in order:
            cell = next(c for c in enabled if c.cell_id == cid)

            # Inter-cell cooldown (첫 cell 제외)
            if results[cid] and not dry_run:
                cooldown(
                    adb, COOLDOWN_MIN_S, COOLDOWN_MAX_S, f"inter-cell {cid}", log_fn
                )

            log_fn(
                f"\n--- {cid} round {r+1} ({'warmup' if is_warmup else 'measure'}) ---"
            )
            tr = run_trial(adb, cell, r, is_warmup, work_dir, log_fn)
            results[cid].append(tr)

            # raw JSON
            tr_dict = asdict(tr)
            with (raw_dir / f"{cid}_round{r:02d}.json").open("w") as f:
                json.dump(tr_dict, f, indent=2)

            # thermal log
            with thermal_log_path.open("a") as tf:
                ts = int(time.time())
                bb = ",".join(
                    f"{tr.thermal_before.get(k, 0):.2f}" for k in ZONES.keys()
                )
                aa = ",".join(
                    f"{tr.thermal_after.get(k, 0):.2f}" for k in ZONES.keys()
                )
                tf.write(f"{ts},before,{cid},{r},{bb}\n")
                tf.write(f"{ts},after,{cid},{r},{aa}\n")

            lat_str = (
                f"{tr.latency_us:.1f} μs" if tr.latency_us is not None else "N/A"
            )
            max_after = (
                max(tr.thermal_after.values()) if tr.thermal_after else 0.0
            )
            log_fn(
                f"  result: ok={tr.ok} lat={lat_str} "
                f"status={tr.measurement_status} "
                f"max_after={max_after:.1f}°C"
            )

            if not dry_run:
                time.sleep(INTER_TRIAL_S)

            if dry_run:
                break  # dry-run: 첫 cell 만 실행

        if dry_run:
            break

        # Inter-round cooldown
        if r < rounds - 1 and not dry_run:
            cooldown(
                adb,
                INTER_ROUND_S,
                INTER_ROUND_MAX_S,
                f"inter-round {r+1}->{r+2}",
                log_fn,
            )

    with seq_path.open("w") as f:
        json.dump(sequence, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def aggregate(
    results: Dict[str, List[TrialResult]],
    cells: List[LcppCell],
    out_dir: Path,
) -> None:
    csv_path = out_dir / "aggregated.csv"
    summary_path = out_dir / "summary.json"

    summary = {}
    with csv_path.open("w") as f:
        f.write(
            "cell_id,backend,op_name,shape_key,dtype,cell_status,"
            "n_raw,n_outlier,n_valid,median_us,median_ms,stdev_us,cv_pct,"
            "p25_us,p75_us,min_us,max_us,measurement_status\n"
        )
        for c in cells:
            if not c.enabled:
                continue
            st = cell_stats(results.get(c.cell_id, []))
            summary[c.cell_id] = {
                "meta": {
                    "backend": c.backend_key,
                    "op": c.op_name,
                    "shape_key": c.shape_key,
                    "dtype": c.dtype,
                    "cell_status": c.cell_status,
                    "desc": c.desc,
                },
                **st,
            }
            f.write(
                ",".join(
                    str(x)
                    for x in [
                        c.cell_id,
                        c.backend_key,
                        c.op_name,
                        c.shape_key,
                        c.dtype,
                        c.cell_status,
                        st.get("n_raw", 0),
                        st.get("n_outlier", 0),
                        st.get("n_valid", 0),
                        st.get("median_us"),
                        st.get("median_ms"),
                        st.get("stdev_us"),
                        st.get("cv_pct"),
                        st.get("p25_us"),
                        st.get("p75_us"),
                        st.get("min_us"),
                        st.get("max_us"),
                        st.get("measurement_status", ""),
                    ]
                )
                + "\n"
            )

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


def generate_report(
    results: Dict[str, List[TrialResult]],
    cells: List[LcppCell],
    out_dir: Path,
    device_serial: str,
    rounds: int,
) -> None:
    md_path = out_dir / "report.md"
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# L.cpp test-backend-ops 측정 결과 (P2)",
        "",
        f"- 생성: {now}",
        f"- device: {device_serial} (Galaxy S25 SM8750, Adreno 830, Hexagon V79)",
        f"- trial protocol: warmup {WARMUP_TRIALS} + measure {MEASURE_TRIALS} (rounds={rounds})",
        f"- model: Qwen 2.5-1.5B (K=1536, FFN=8960, vocab=151936, GQA n_kv=2)",
        f"- backends: lcpp.cpu (CPU), lcpp.gpu (GPUOpenCL), lcpp.htp (HTP0 — device_init_failed)",
        "",
        "## 측정 결과 매트릭스",
        "",
        "| cell_id | backend | op | shape | dtype | status | n_valid | median (μs) | CV (%) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for c in cells:
        if not c.enabled:
            continue
        st = cell_stats(results.get(c.cell_id, []))
        cv = st.get("cv_pct")
        median = st.get("median_us")

        if st.get("n_valid", 0) == 0:
            cv_str = "-"
            med_str = st.get("measurement_status", "-")
        else:
            cv_str = f"{cv:.1f}" if cv is not None else "-"
            med_str = f"{median:.1f}" if median is not None else "-"

        lines.append(
            f"| {c.cell_id} | {c.backend_key} | {c.op_name} | {c.shape_key} | "
            f"{c.dtype} | {c.cell_status} | {st.get('n_valid', 0)}/{st.get('n_raw', 0)} | "
            f"{med_str} | {cv_str} |"
        )

    lines += [
        "",
        "## lcpp.htp 상태",
        "",
        "- HTP0 device session 생성 실패: `ggml-hex: FastRPC capability query failed (err -1)` + `failed to create device/session 0`",
        "- S25 SM8750 에서 unsigned PD 허용되지 않음: `failed to enable unsigned PD for session 0 : error 0xffffffff`",
        "- ggml-hexagon.cpp 가 v73으로 fallback 하지만 S25 는 V79 — 버전 불일치",
        "- 모든 perf cell: `⚠ device_init_failed`",
        "- graph-level 참조 (llama-bench HTP0 tg32 = 32.40 tok/s, v1 결과 inherit)",
        "",
        "## perf case 없는 op 목록",
        "",
        "다음 op 는 test-backend-ops 에 perf test case 정의 없음 (support/test 모드만 가능):",
        "- SILU (UNARY), MUL, RMS_NORM, GET_ROWS, SCALE, SET_ROWS, FLASH_ATTN_EXT",
        "- 이 op 들의 latency = null, measurement_status = no_perf_case",
        "",
        "## 기술 노트",
        "",
        "- LD_PRELOAD: `libggml-hexagon-stub.so` (HTP0 device_count=0 override)",
        "  - libggml-hexagon.so 는 정적 링크 (NEEDED) 로 제거 불가",
        "  - ggml_backend_dev_get(1) 가 NULL 반환 시 GGML_ASSERT crash → stub 으로 우회",
        "  - CPU/GPU 단독 실행 시 필수",
        "- -b GPUOpenCL, -b CPU: 정확한 backend 이름 (ggml_backend_dev_name() 반환값)",
        "- 출력 parsing: `<N> runs - <latency> us/run` 패턴 (regex)",
    ]

    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Push binaries to device
# ---------------------------------------------------------------------------

def push_binaries(adb: Adb, log_fn, force: bool = False) -> bool:
    """test-backend-ops + 의존성 .so + stub push."""
    binary_path = HOST_LCPP_DIR / LCPP_BINARY
    if not binary_path.exists():
        log_fn(f"[push] MISSING: {binary_path}")
        return False

    # 이미 있는지 확인
    existing = adb.shell(
        f"test -x {DEVICE_WORK_DIR}/{LCPP_BINARY} && echo yes || echo no",
        check=False,
    ).strip()

    if existing == "yes" and not force:
        log_fn(f"[push] {LCPP_BINARY} already on device (use --force-push to re-push)")
    else:
        log_fn(f"[push] {binary_path} → {DEVICE_WORK_DIR}/")
        adb.push(str(binary_path), f"{DEVICE_WORK_DIR}/")
        adb.shell(f"chmod +x {DEVICE_WORK_DIR}/{LCPP_BINARY}")

    # .so 의존성
    so_files = [
        "libggml-base.so",
        "libggml-cpu.so",
        "libggml-opencl.so",
        "libggml-hexagon.so",
        "libggml.so",
        "libllama.so",
    ]
    for so in so_files:
        src = HOST_LCPP_DIR / so
        if not src.exists():
            log_fn(f"[push] WARN: {so} not found at {src}")
            continue
        dst_check = adb.shell(
            f"test -f {DEVICE_WORK_DIR}/{so} && echo yes || echo no", check=False
        ).strip()
        if dst_check == "yes" and not force:
            pass
        else:
            log_fn(f"[push] {so}")
            adb.push(str(src), f"{DEVICE_WORK_DIR}/")

    # libggml-htp-v79.so (Hexagon DSP skel, V79 = SM8750)
    htp_so_src = (
        HOST_LCPP_DIR.parent.parent
        / "ggml/src/ggml-hexagon/libggml-htp-v79.so"
    )
    if htp_so_src.exists():
        adb.push(str(htp_so_src), f"{DEVICE_WORK_DIR}/")
        log_fn(f"[push] libggml-htp-v79.so")
    else:
        log_fn(f"[push] WARN: libggml-htp-v79.so not found at {htp_so_src}")

    # stub .so
    if not HOST_STUB_SO.exists():
        log_fn(f"[push] WARN: stub .so not found at {HOST_STUB_SO} — building...")
        _build_stub_so(log_fn)

    if HOST_STUB_SO.exists():
        adb.push(str(HOST_STUB_SO), f"{DEVICE_WORK_DIR}/{LCPP_STUB_SO}")
        log_fn(f"[push] {LCPP_STUB_SO} (HTP crash stub)")
    else:
        log_fn(f"[push] ERROR: stub .so build failed — CPU/GPU measurement will crash")
        return False

    return True


def _build_stub_so(log_fn) -> None:
    """libggml-hexagon stub .so 빌드 (NDK clang 사용)."""
    ndk_base = Path("/opt/android-ndk")
    clang = ndk_base / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang"
    if not clang.exists():
        # 최신 API version 탐색
        bin_dir = ndk_base / "toolchains/llvm/prebuilt/linux-x86_64/bin"
        candidates = sorted(bin_dir.glob("aarch64-linux-android*-clang"))
        if candidates:
            clang = candidates[-1]
        else:
            log_fn(f"[stub-build] ERROR: NDK clang not found at {ndk_base}")
            return

    src = "/tmp/libggml_hexagon_stub.c"
    stub_c = """
#include <stddef.h>
typedef void* ggml_backend_reg_t;
typedef void* ggml_backend_dev_t;
typedef struct {
    const char* (*get_name)(ggml_backend_reg_t);
    size_t (*get_device_count)(ggml_backend_reg_t);
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t, size_t);
    void* (*get_proc_address)(ggml_backend_reg_t, const char*);
} ggml_backend_reg_iface_t;
typedef struct { ggml_backend_reg_iface_t iface; void* context; } ggml_backend_reg_s;
static const char* _name(ggml_backend_reg_t r) { (void)r; return "HTP"; }
static size_t _count(ggml_backend_reg_t r) { (void)r; return 0; }
static ggml_backend_dev_t _dev(ggml_backend_reg_t r, size_t i) { (void)r; (void)i; return NULL; }
static void* _proc(ggml_backend_reg_t r, const char* n) { (void)r; (void)n; return NULL; }
static ggml_backend_reg_s _stub = { .iface = { _name, _count, _dev, _proc }, .context = NULL };
ggml_backend_reg_t ggml_backend_hexagon_reg(void) { return (ggml_backend_reg_t)&_stub; }
"""
    Path(src).write_text(stub_c)
    r = subprocess.run(
        [str(clang), "-shared", "-fPIC", "-o", str(HOST_STUB_SO), src],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log_fn(f"[stub-build] build failed: {r.stderr[:200]}")
    else:
        log_fn(f"[stub-build] built {HOST_STUB_SO}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="L.cpp test-backend-ops same-device driver (P2)"
    )
    parser.add_argument(
        "--device", "--serial", default="R3CY408S5SB",
        dest="serial",
        help="ADB device serial (default: R3CY408S5SB = Galaxy S25)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--backends", default="lcpp.cpu,lcpp.gpu,lcpp.htp",
        help="Comma-separated backends (lcpp.cpu,lcpp.gpu,lcpp.htp)",
    )
    parser.add_argument(
        "--ops", default="",
        help="Comma-separated op names to measure (empty = all 12 ops)",
    )
    parser.add_argument(
        "--dtypes", default="f16,q4_0",
        help="Comma-separated dtypes (f16,q4_0)",
    )
    parser.add_argument(
        "--rounds", type=int, default=ROUNDS,
        help=f"Total rounds (default {ROUNDS} = {WARMUP_TRIALS}w + {MEASURE_TRIALS}m)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="1-cell × 1-round smoke test (no cooldown)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="단축 cooldown (inter-cell 10s, inter-round 30s) — protocol 검증용",
    )
    parser.add_argument(
        "--force-push", action="store_true",
        help="Always re-push binaries",
    )
    parser.add_argument(
        "--skip-push", action="store_true",
        help="Skip binary push (already on device)",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Only verify ADB + thermal + binaries, then exit",
    )
    args = parser.parse_args()

    if args.quick:
        global COOLDOWN_MIN_S, COOLDOWN_MAX_S, INTER_ROUND_S, INTER_ROUND_MAX_S
        global SESSION_WARMUP_S
        COOLDOWN_MIN_S = 10
        COOLDOWN_MAX_S = 60
        INTER_ROUND_S = 30
        INTER_ROUND_MAX_S = 90
        SESSION_WARMUP_S = 15

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "driver.log"
    log_f = log_path.open("w")

    def log_fn(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log_fn(f"[init] device={args.serial}")
    log_fn(f"[init] out={out_dir}")
    log_fn(f"[init] dry_run={args.dry_run}")

    # Build cell list
    backends_filter = [b.strip() for b in args.backends.split(",") if b.strip()]
    ops_filter = [o.strip() for o in args.ops.split(",") if o.strip()] or None
    dtypes_filter = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    cells = build_cells(
        ops_filter=ops_filter,
        backends_filter=backends_filter,
        dtypes_filter=dtypes_filter,
    )
    log_fn(f"[init] total cells: {len(cells)}")
    measurable = [c for c in cells if c.is_measurable()]
    no_perf = [c for c in cells if c.enabled and not c.is_measurable()]
    log_fn(f"[init] measurable (latency): {len(measurable)}")
    log_fn(f"[init] no-perf / device-fail: {len(no_perf)}")

    adb = Adb(args.serial)

    # Preflight: zombie check
    zombies = adb.zombie_check()
    if zombies:
        log_fn("[preflight] FAIL — zombies:")
        for z in zombies:
            log_fn(f"  {z}")
        return 3

    # Preflight: thermal
    name, t = adb.max_zone_temp()
    log_fn(f"[preflight] max_zone={name}={t:.1f}°C (start<{START_TEMP_C}°C)")
    if t > START_TEMP_C:
        log_fn("[preflight] too hot — cooling down")
        if not wait_thermal_below(adb, START_TEMP_C, 300, "preflight", log_fn):
            log_fn("[preflight] cooldown timeout — abort")
            return 4

    # Binary push
    if not args.skip_push:
        ok = push_binaries(adb, log_fn, force=args.force_push)
        if not ok:
            log_fn("[push] FAILED — binary push error")
            return 5
    else:
        log_fn("[push] skipped (--skip-push)")

    # Verify binary on device
    exists = adb.shell(
        f"test -x {DEVICE_WORK_DIR}/{LCPP_BINARY} && echo yes || echo no", check=False
    ).strip()
    if exists != "yes":
        log_fn(f"[verify] {LCPP_BINARY} NOT found on device")
        return 6
    log_fn(f"[verify] {LCPP_BINARY} on device OK")

    stub_exists = adb.shell(
        f"test -f {DEVICE_WORK_DIR}/{LCPP_STUB_SO} && echo yes || echo no", check=False
    ).strip()
    log_fn(f"[verify] {LCPP_STUB_SO}: {stub_exists}")

    if args.preflight_only:
        log_fn("[preflight-only] OK — exiting")
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
        "trigger_temp_c": TRIGGER_TEMP_C,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": args.dry_run,
        "backends": backends_filter,
        "ops_filter": ops_filter,
        "dtypes": dtypes_filter,
        "total_cells": len(cells),
        "measurable_cells": len(measurable),
        "lcpp_binary": LCPP_BINARY,
        "stub_so": LCPP_STUB_SO,
        "note": "lcpp.htp = device_init_failed on S25 SM8750 (unsigned PD error)",
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))

    # Run
    try:
        results = run_matrix(
            adb, cells, out_dir, args.rounds, args.seed, log_fn,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        log_fn("[interrupted]")
        log_f.close()
        return 130
    except Exception as e:
        log_fn(f"[error] {type(e).__name__}: {e}")
        log_f.close()
        raise

    aggregate(results, cells, out_dir)
    generate_report(results, cells, out_dir, args.serial, args.rounds)

    log_fn("\n[done] outputs:")
    log_fn(f"  {out_dir}/raw/")
    log_fn(f"  {out_dir}/aggregated.csv")
    log_fn(f"  {out_dir}/summary.json")
    log_fn(f"  {out_dir}/thermal_log.csv")
    log_fn(f"  {out_dir}/report.md")
    log_fn(f"  {out_dir}/env.json")

    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
