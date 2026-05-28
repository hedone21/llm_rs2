#!/usr/bin/env python3
"""microbench_qnn_matrix.py — QNN HTP 비교 측정 driver.

두 가지 모드:
  1. 기본 (legacy): 9-cell + Ref 매트릭스 (μ-Q1 4-cell backward-compat)
  2. full matrix: --matrix full → 7-col × 13-op × 2-dtype 196 cell (P3 산출물)

full matrix cell_id 명명 (matrix.md v2-§8):
  <backend>_<op>_<dtype>[_<shape_id>]
  예: ours_cpu_MUL_MAT_f16_mm_ffn, lcpp_htp_FLASH_ATTN_EXT_f16

full matrix backend 7개:
  ours.cpu   — ARM64 NEON CPU (microbench_neon_op_matrix)
  ours.gpu   — OpenCL Adreno 830 (microbench_opencl_op_matrix)
  ours.htp   — Ours HTP FastRPC NPU (microbench_htp_<op>[_f16])
  et.htp     — ExecuTorch QNN HTP (qnn_executor_runner + .pte)
  lcpp.cpu   — llama.cpp ARM64 CPU (test-backend-ops -b CPU)
  lcpp.gpu   — llama.cpp OpenCL Adreno (test-backend-ops -b GPUOpenCL)
  lcpp.htp   — llama.cpp ggml-hexagon HTP0 (test-backend-ops -b HTP0)

status 5-tier (matrix.md v2-§5.5):
  ✓   existing (μ-Q1 inherit)
  +   new bin (측정 대상)
  ⚠   perf abort / device_init_failed / no_perf_case
  ✗   hard 미지원
  —   fair 비교 부적합 (activation-only Q4_0 row 등)

핵심 protocol (plan_qnn_microbench_measurement_protocol_2026_05_26.md):
- warmup 3 + measure 10 trial per cell
- round-robin shuffle (cell 순서 발열 누적 편향 제거)
- 8 thermal zone (CPU little/mid/prime + GPU + Hexagon HVX/HMX + DDR) polling
- 50°C trigger / 60s session warmup / 45~180s inter-cell cooldown
- Tukey 1.5×IQR outlier rejection
- raw + aggregated.csv + thermal_log.csv + report.md + summary.json 출력
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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
DEVICE_BIN_DIR = "/data/local/tmp"
DEVICE_QNN_LIBDIR = "/data/local/tmp/qnn"
DEVICE_WORK_DIR = "/data/local/tmp"
DEVICE_EXECUTORCH_DIR = "/data/local/tmp/executorch"
# D2 fix: HTP0 selective disable stub (CPU/GPU 단독 실행 시 unsigned PD error 방지)
LCPP_STUB_SO = "libggml-hexagon-stub.so"

HOST_LCPP_DIR = Path("/home/go/Workspace/llama.cpp/build-snapdragon/bin")

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


# ==============================================================================
# ADB wrapper
# ==============================================================================

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
            "ps -A 2>/dev/null | grep -E '(microbench|generate|qnn_executor|llama-cli|test-backend)' | grep -v grep",
            check=False,
        )
        return [l.strip() for l in out.splitlines() if l.strip()]

    def file_exists(self, path: str) -> bool:
        r = self.shell(f"test -f {path} && echo yes || echo no", check=False).strip()
        return r == "yes"


# ==============================================================================
# Legacy Cell registry (μ-Q1 9-cell backward-compat)
# ==============================================================================

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
    group_key: Optional[str] = None
    latency_pattern: Optional[str] = None
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
        ld_parts = [wd] + self.extra_ld_paths + [DEVICE_QNN_LIBDIR, "/system/lib64"]
        ld_path = ":".join(ld_parts)
        adsp = f"ADSP_LIBRARY_PATH={wd} "
        return (
            f"cd {wd} && "
            f"LD_LIBRARY_PATH={ld_path} {adsp}{env_prefix}"
            f"taskset 3f {bin_path} {args_str}"
        )


def build_cells(enable: List[str]) -> List[Cell]:
    n_iters_str = str(MEASURE_TRIALS)
    cells = [
        Cell("M1",  "fp32", "htp",
             "microbench_htp_matmul_correctness",
             args=[str(K), str(N)],
             tolerance_max_abs=1e-3, tolerance_cosine=0.9999),
        Cell("M1b", "f16",  "htp",
             "microbench_htp_matmul_correctness",
             args=[str(K), str(N), "f16"],
             tolerance_max_abs=1e-2, tolerance_cosine=0.999,
             optional=True),
        Cell("M2",  "w8a8", "htp",
             "microbench_htp_matmul_w8a8",
             args=[str(K), str(N)],
             tolerance_max_abs=0.1, tolerance_cosine=0.99,
             optional=True),
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
        Cell("M5",  "q4_0", "opencl",
             "microbench_qnn_oppkg_matmul_q40_correct",
             args=[],
             env={"QNN_BACKEND_LIB": f"{DEVICE_QNN_LIBDIR}/libQnnGpu.so"},
             tolerance_max_abs=0.05, tolerance_cosine=0.999,
             optional=True),
        Cell("M6b", "f16",  "executorch",
             "qnn_executor_runner",
             args=["--model_path", "matmul_f16.pte",
                   "--input_list_path", "input_list.txt",
                   "--warm_up", "3", "--iteration", "10"],
             tolerance_max_abs=1e-2, tolerance_cosine=0.999,
             work_dir_override=DEVICE_EXECUTORCH_DIR,
             bin_dir_override=DEVICE_EXECUTORCH_DIR,
             latency_pattern=r"inference took [\d.]+\s*ms,\s*avg\s+([\d.]+)\s*ms"),
        Cell("M7",  "w8a8", "executorch",
             "qnn_executor_runner",
             args=["--model_path", "matmul_w8a8.pte",
                   "--input_list_path", "input_list.txt",
                   "--warm_up", "3", "--iteration", "10"],
             tolerance_max_abs=0.1, tolerance_cosine=0.99,
             work_dir_override=DEVICE_EXECUTORCH_DIR,
             bin_dir_override=DEVICE_EXECUTORCH_DIR,
             latency_pattern=r"inference took [\d.]+\s*ms,\s*avg\s+([\d.]+)\s*ms"),
        Cell("Ref", "f32",  "cpu-neon",
             "microbench_cpu_gemv_f32_ref",
             args=[str(K), str(N), "6"],
             tolerance_max_abs=0.0, tolerance_cosine=1.0,
             optional=True),
    ]
    if enable:
        names = set(enable)
        for c in cells:
            c.enabled = c.name in names
    else:
        default_off = {"M1", "Ref"}
        for c in cells:
            c.enabled = (not c.optional) and (c.name not in default_off)
    return cells


# ==============================================================================
# Full Matrix Cell registry (P3, 196 cell)
# ==============================================================================

# Qwen 2.5-1.5B 모델 상수
QWEN = {
    "hidden": 1536,
    "ffn": 8960,
    "n_heads_q": 12,
    "n_heads_kv": 2,
    "head_dim": 128,
    "vocab": 151936,
    "ctx": 1024,
}

# MUL_MAT 3 shape 정의 (matrix.md v2-§3)
MULMAT_SHAPES = {
    "mm_ffn": {"K": 1536, "N": 8960,   "desc": "FFN gate/up GEMV"},
    "mm_lmh": {"K": 1536, "N": 151936, "desc": "LM head GEMV"},
    "mm_qkv": {"K": 1536, "N": 2048,   "desc": "QKV fused GEMV (Q=1536+K=256+V=256)"},
}

# 13 op 목록 (Tier A 8 + Tier B 4 + Tier D 1 SWIGLU 제외됨 = 12, 본 문서에선 SET_ROWS 포함 12)
# matrix.md v2-§2 기준
ALL_OPS = [
    "MUL_MAT", "RMS_NORM", "ROPE", "FLASH_ATTN_EXT", "GET_ROWS",
    "SILU", "MUL", "ADD",          # Tier A
    "SOFT_MAX", "SCALE", "CPY", "SET_ROWS",  # Tier B
]

# activation-only op: Q4_0 row는 fair 부적합 → — 마킹 (matrix.md v2-§5.1)
ACTIVATION_ONLY_OPS = {
    "RMS_NORM", "ROPE", "FLASH_ATTN_EXT", "SILU", "MUL", "ADD",
    "SOFT_MAX", "SCALE", "CPY", "SET_ROWS",
}

# GET_ROWS Q4_0 row: embed Q4 측정 (embed table이 Q4일 때) — 측정 대상
# MUL_MAT Q4_0 row: Q4_0 weight → 측정 대상


@dataclass
class FullCell:
    """full matrix 196 cell 단위."""
    cell_id: str
    backend: str      # ours.cpu / ours.gpu / ours.htp / et.htp / lcpp.cpu / lcpp.gpu / lcpp.htp
    op: str
    dtype: str        # f16 / q4_0
    shape_id: str     # mm_ffn / mm_lmh / mm_qkv / (op 이름 그대로)
    shape_desc: str
    status: str       # ✓ / + / ⚠ / ✗ / —
    proxy_note: str = ""  # P1a 발견 proxy 측정 패턴
    # ADB 실행 정보
    bin_name: str = ""
    bin_dir: str = DEVICE_BIN_DIR
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    extra_ld_paths: List[str] = field(default_factory=list)
    work_dir: str = DEVICE_WORK_DIR
    latency_pattern: Optional[str] = None  # stdout regex (1 capture group, ms 단위)
    latency_unit: str = "ms"  # ms / us / ns
    # tolerance
    tol_max_abs: float = 1e-2
    tol_cosine: float = 0.999

    def is_measurable(self) -> bool:
        """실제 측정 시도 여부 (✓/+ 만 true, ✗/⚠/— 는 false)."""
        return self.status in ("✓", "+")

    def adb_cmd(self) -> str:
        env_str = " ".join(f"{k}={v}" for k, v in self.env.items())
        env_prefix = f"{env_str} " if env_str else ""
        ld_parts = [self.work_dir] + self.extra_ld_paths + [DEVICE_QNN_LIBDIR, "/system/lib64"]
        ld_path = ":".join(ld_parts)
        adsp = f"ADSP_LIBRARY_PATH={self.work_dir} "
        bin_path = f"{self.bin_dir}/{self.bin_name}"
        args_str = " ".join(self.args)
        return (
            f"cd {self.work_dir} && "
            f"LD_LIBRARY_PATH={ld_path} {adsp}{env_prefix}"
            f"taskset 3f {bin_path} {args_str}"
        )


def _cell_id(backend: str, op: str, dtype: str, shape_id: str) -> str:
    """cell_id 생성 (matrix.md v2-§8 convention)."""
    b = backend.replace(".", "_")
    if shape_id and shape_id != op:
        return f"{b}_{op}_{dtype}_{shape_id}"
    return f"{b}_{op}_{dtype}"


def _dash_cell(backend: str, op: str, dtype: str, shape_id: str) -> FullCell:
    """— (fair 부적합) cell: 목록에 포함되지만 측정 안 함."""
    return FullCell(
        cell_id=_cell_id(backend, op, dtype, shape_id),
        backend=backend,
        op=op, dtype=dtype, shape_id=shape_id,
        shape_desc="activation-only op: Q4_0 row fair 부적합",
        status="—",
        proxy_note="fair_pair_incompatible",
    )


def _ours_cpu_cells() -> List[FullCell]:
    """ours.cpu — microbench_neon_op_matrix CLI."""
    cells = []
    for dtype in ("f16", "q4_0"):
        for op in ALL_OPS:
            # activation-only Q4_0 row = — (fair 부적합)
            if dtype == "q4_0" and op in ACTIVATION_ONLY_OPS:
                # GET_ROWS Q4_0은 embed Q4 측정 가능 → 측정 대상
                if op != "GET_ROWS":
                    if op == "MUL_MAT":
                        for sid in MULMAT_SHAPES:
                            cells.append(_dash_cell("ours.cpu", op, dtype, sid))
                    else:
                        cells.append(_dash_cell("ours.cpu", op, dtype, op))
                    continue

            if op == "MUL_MAT":
                for sid, sdef in MULMAT_SHAPES.items():
                    proxy = ""
                    status = "+"
                    cells.append(FullCell(
                        cell_id=_cell_id("ours.cpu", op, dtype, sid),
                        backend="ours.cpu",
                        op=op, dtype=dtype, shape_id=sid,
                        shape_desc=sdef["desc"],
                        status=status,
                        proxy_note=proxy,
                        bin_name="microbench_neon_op_matrix",
                        bin_dir=DEVICE_BIN_DIR,
                        args=["--op", "MUL_MAT", "--dtype", dtype, "--shape", sid,
                              "--warmup", "3", "--measure", "10"],
                        latency_pattern=r'"median_ns"\s*:\s*([\d.]+)',
                        latency_unit="ns",
                        tol_max_abs=5e-2 if dtype == "q4_0" else 1e-2,
                    ))
            else:
                proxy = ""
                # P1a proxy: MUL_f16 = add_assign proxy
                if op == "MUL" and dtype == "f16":
                    proxy = "add_assign proxy"
                status = "+"
                cells.append(FullCell(
                    cell_id=_cell_id("ours.cpu", op, dtype, op),
                    backend="ours.cpu",
                    op=op, dtype=dtype, shape_id=op,
                    shape_desc=_op_shape_desc(op),
                    status=status,
                    proxy_note=proxy,
                    bin_name="microbench_neon_op_matrix",
                    bin_dir=DEVICE_BIN_DIR,
                    args=["--op", op, "--dtype", dtype,
                          "--warmup", "3", "--measure", "10"],
                    latency_pattern=r'"median_ns"\s*:\s*([\d.]+)',
                    latency_unit="ns",
                    tol_max_abs=1e-2,
                ))
    return cells


def _ours_gpu_cells() -> List[FullCell]:
    """ours.gpu — microbench_opencl_op_matrix CLI."""
    cells = []
    for dtype in ("f16", "q4_0"):
        for op in ALL_OPS:
            if dtype == "q4_0" and op in ACTIVATION_ONLY_OPS:
                if op != "GET_ROWS":
                    if op == "MUL_MAT":
                        for sid in MULMAT_SHAPES:
                            cells.append(_dash_cell("ours.gpu", op, dtype, sid))
                    else:
                        cells.append(_dash_cell("ours.gpu", op, dtype, op))
                    continue

            if op == "MUL_MAT":
                for sid, sdef in MULMAT_SHAPES.items():
                    cells.append(FullCell(
                        cell_id=_cell_id("ours.gpu", op, dtype, sid),
                        backend="ours.gpu",
                        op=op, dtype=dtype, shape_id=sid,
                        shape_desc=sdef["desc"],
                        status="+",
                        bin_name="microbench_opencl_op_matrix",
                        bin_dir=DEVICE_BIN_DIR,
                        args=["--ops", f"MUL_MAT_{dtype.upper()}_{sid.upper()}"],
                        latency_pattern=r'"median_ns"\s*:\s*([\d.]+)',
                        latency_unit="ns",
                        tol_max_abs=5e-2 if dtype == "q4_0" else 1e-2,
                    ))
            else:
                cells.append(FullCell(
                    cell_id=_cell_id("ours.gpu", op, dtype, op),
                    backend="ours.gpu",
                    op=op, dtype=dtype, shape_id=op,
                    shape_desc=_op_shape_desc(op),
                    status="+",
                    bin_name="microbench_opencl_op_matrix",
                    bin_dir=DEVICE_BIN_DIR,
                    args=["--ops", f"{op}_{dtype.upper()}"],
                    latency_pattern=r'"median_ns"\s*:\s*([\d.]+)',
                    latency_unit="ns",
                    tol_max_abs=1e-2,
                ))
    return cells


# P1d 발견: Ours-NPU F16 DSP-side 비대칭 지원 (sprint master §P1d)
_OURS_HTP_F16_SUPPORT: Dict[str, str] = {
    # op → status  (+: GREEN attempt, ✗: NO_SUPPORT)
    "MUL_MAT":        "+",   # matmul-ops.c f16-f32 path
    "ADD":            "+",   # binary-ops.c hvx_add_f16_*
    "MUL":            "+",   # binary-ops.c hvx_mul_f16_*
    "FLASH_ATTN_EXT": "+",   # flash-attn-ops.c:618 Q∈{F16,F32} attempt
    "SILU":           "✗",   # unary-ops.c F32 only → NO_SUPPORT
    "RMS_NORM":       "✗",
    "ROPE":           "✗",
    "GET_ROWS":       "✗",
    "SOFT_MAX":       "✗",
    "SCALE":          "✗",   # init_scale_req 미구현
    "CPY":            "✗",   # init_cpy_req 미구현
    "SET_ROWS":       "✗",   # init_set_rows_req 미구현
}

# P1d 발견: Q4_0 Tier-A 8/8 GREEN (G1~G5)
_OURS_HTP_Q4_SUPPORT: Dict[str, str] = {
    "MUL_MAT":        "✓",  # G2 inherit
    "RMS_NORM":       "✓",  # G1 inherit
    "ROPE":           "✓",  # G3 inherit
    "FLASH_ATTN_EXT": "✗",  # NPU fused FA 미지원
    "GET_ROWS":       "✓",  # G6 inherit (vocab=1024 dummy, full vocab P1d gate 대기)
    "SILU":           "✓",  # G4 inherit
    "MUL":            "✓",  # G4 inherit
    "ADD":            "✓",  # G4 inherit
    "SOFT_MAX":       "✓",  # G5 inherit
    "SCALE":          "✗",
    "CPY":            "✗",
    "SET_ROWS":       "✗",
}

# P1d F16 bin 파일 매핑 (작성된 5개 bin)
_OURS_HTP_F16_BIN: Dict[str, str] = {
    "MUL_MAT":        "microbench_htp_matmul_f16",
    "ADD":            "microbench_htp_add_f16",
    "MUL":            "microbench_htp_mul_f16",
    "FLASH_ATTN_EXT": "microbench_htp_flash_attn_ext",
    "SILU":           "microbench_htp_silu_f16",  # NO_SUPPORT evidence
}

# Q4_0 bin 매핑 (G1~G6 기존 bin)
_OURS_HTP_Q4_BIN: Dict[str, str] = {
    "MUL_MAT":        "microbench_htp_matmul",
    "RMS_NORM":       "microbench_htp_rmsnorm",
    "ROPE":           "microbench_htp_rope",
    "GET_ROWS":       "microbench_htp_get_rows",
    "SILU":           "microbench_htp_silu",
    "MUL":            "microbench_htp_mul",
    "ADD":            "microbench_htp_add",
    "SOFT_MAX":       "microbench_htp_softmax",
}


def _ours_htp_cells() -> List[FullCell]:
    """ours.htp — microbench_htp_<op>[_f16] bin."""
    cells = []
    for dtype in ("f16", "q4_0"):
        support_map = _OURS_HTP_F16_SUPPORT if dtype == "f16" else _OURS_HTP_Q4_SUPPORT
        bin_map = _OURS_HTP_F16_BIN if dtype == "f16" else _OURS_HTP_Q4_BIN

        for op in ALL_OPS:
            # activation-only Q4_0 = —
            if dtype == "q4_0" and op in ACTIVATION_ONLY_OPS:
                if op != "GET_ROWS":
                    if op == "MUL_MAT":
                        for sid in MULMAT_SHAPES:
                            cells.append(_dash_cell("ours.htp", op, dtype, sid))
                    else:
                        cells.append(_dash_cell("ours.htp", op, dtype, op))
                    continue

            status = support_map.get(op, "✗")

            if op == "MUL_MAT":
                for sid in MULMAT_SHAPES:
                    # mm_lmh N=151936 + mm_qkv N=2048: P1d gate 대기 중
                    if dtype == "f16" and status == "+":
                        cell_status = "+"
                    elif dtype == "q4_0":
                        # G2 승계: mm_ffn만 검증됨. mm_lmh/mm_qkv는 P1d gate
                        cell_status = "+" if sid == "mm_ffn" else "+"
                    else:
                        cell_status = status
                    b = bin_map.get(op, "")
                    cells.append(FullCell(
                        cell_id=_cell_id("ours.htp", op, dtype, sid),
                        backend="ours.htp",
                        op=op, dtype=dtype, shape_id=sid,
                        shape_desc=MULMAT_SHAPES[sid]["desc"],
                        status=cell_status,
                        bin_name=b,
                        bin_dir=DEVICE_BIN_DIR,
                        args=["--shape", sid, "--warmup", "3", "--measure", "10"],
                        latency_pattern=r'"median_ns"\s*:\s*([\d.]+)',
                        latency_unit="ns",
                        tol_max_abs=5e-2 if dtype == "q4_0" else 1e-2,
                        extra_ld_paths=["/data/local/tmp"],
                    ))
            else:
                b = bin_map.get(op, "")
                cells.append(FullCell(
                    cell_id=_cell_id("ours.htp", op, dtype, op),
                    backend="ours.htp",
                    op=op, dtype=dtype, shape_id=op,
                    shape_desc=_op_shape_desc(op),
                    status=status,
                    bin_name=b,
                    bin_dir=DEVICE_BIN_DIR,
                    args=["--warmup", "3", "--measure", "10"],
                    latency_pattern=r'"median_ns"\s*:\s*([\d.]+)',
                    latency_unit="ns",
                    tol_max_abs=1e-2,
                    extra_ld_paths=["/data/local/tmp"],
                ))
    return cells


# P1c 발견: ExecuTorch .pte 결과 (build_pte_sweep.py 산출물)
_ET_HTP_PTE_STATUS: Dict[str, str] = {
    # (op, shape_id, dtype) → status
    # FLASH_ATTN_EXT: Dynamo FX export 실패 → ✗
    # CPY: aten._to_copy delegate 실패 → ✗
    # GET_ROWS vocab=151936: OOM → vocab=1024 fallback, paper annotation 필요
}

_ET_HTP_HARD_FAIL_OPS = {"FLASH_ATTN_EXT", "CPY"}  # P1c ✗ 확정


def _pte_path(op: str, shape_id: str, dtype: str) -> str:
    """device 상의 .pte 경로."""
    dtype_tag = "fp16" if dtype == "f16" else "w4a8"
    if op == "MUL_MAT":
        return f"{DEVICE_EXECUTORCH_DIR}/pte/mul_mat_{shape_id}_{dtype_tag}.pte"
    op_lower = op.lower()
    return f"{DEVICE_EXECUTORCH_DIR}/pte/{op_lower}_{dtype_tag}.pte"


def _et_htp_cells() -> List[FullCell]:
    """et.htp — ExecuTorch qnn_executor_runner + .pte."""
    cells = []
    for dtype in ("f16", "q4_0"):
        dtype_tag = "fp16" if dtype == "f16" else "w4a8"

        for op in ALL_OPS:
            if dtype == "q4_0" and op in ACTIVATION_ONLY_OPS:
                if op != "GET_ROWS":
                    if op == "MUL_MAT":
                        for sid in MULMAT_SHAPES:
                            cells.append(_dash_cell("et.htp", op, dtype, sid))
                    else:
                        cells.append(_dash_cell("et.htp", op, dtype, op))
                    continue

            # P1c hard fail
            if op in _ET_HTP_HARD_FAIL_OPS:
                status = "✗"
            else:
                status = "+"

            if op == "MUL_MAT":
                for sid, sdef in MULMAT_SHAPES.items():
                    # mm_lmh vocab=151936: OOM 위험 → paper annotation 필요
                    pte = _pte_path(op, sid, dtype)
                    cells.append(FullCell(
                        cell_id=_cell_id("et.htp", op, dtype, sid),
                        backend="et.htp",
                        op=op, dtype=dtype, shape_id=sid,
                        shape_desc=sdef["desc"],
                        status=status,
                        proxy_note="vocab=1024 fallback" if sid == "mm_lmh" else "",
                        bin_name="qnn_executor_runner",
                        bin_dir=DEVICE_EXECUTORCH_DIR,
                        args=["--model_path", pte,
                              "--warm_up", "3", "--iteration", "10"],
                        work_dir=DEVICE_EXECUTORCH_DIR,
                        latency_pattern=r"avg\s+([\d.]+)\s*ms",
                        latency_unit="ms",
                        tol_max_abs=0.15 if sid == "mm_lmh" else 0.1,
                        tol_cosine=0.99,
                    ))
            else:
                pte = _pte_path(op, op, dtype)
                cells.append(FullCell(
                    cell_id=_cell_id("et.htp", op, dtype, op),
                    backend="et.htp",
                    op=op, dtype=dtype, shape_id=op,
                    shape_desc=_op_shape_desc(op),
                    status=status,
                    bin_name="qnn_executor_runner",
                    bin_dir=DEVICE_EXECUTORCH_DIR,
                    args=["--model_path", pte,
                          "--warm_up", "3", "--iteration", "10"],
                    work_dir=DEVICE_EXECUTORCH_DIR,
                    latency_pattern=r"avg\s+([\d.]+)\s*ms",
                    latency_unit="ms",
                    tol_max_abs=1e-2,
                    tol_cosine=0.99,
                ))
    return cells


# P2 발견: lcpp perf case 없는 op
_LCPP_NO_PERF_CASE = {"SILU", "MUL", "RMS_NORM", "GET_ROWS", "SCALE", "SET_ROWS", "FLASH_ATTN_EXT"}
# P2 발견: F16 dtype 없는 op (F32 fallback만)
_LCPP_F16_F32_FALLBACK = {"ADD", "ROPE", "SOFT_MAX", "CPY"}
# P2 발견: lcpp.htp device_init_failed
_LCPP_HTP_DEVICE_FAIL = True

# lcpp MUL_MAT 미지원 backend cell 상황: shape_mismatch
# (test-backend-ops 고정 m=4096,k=14336, Qwen shape 없음)
_LCPP_MULMAT_SHAPE_NOTE = "shape_mismatch: test-backend-ops fixes m=4096,k=14336; Qwen shapes unavailable"


def _lcpp_cells(backend_key: str) -> List[FullCell]:
    """lcpp.{cpu,gpu,htp} — test-backend-ops wrapper."""
    cells = []
    flag = {"lcpp.cpu": "CPU", "lcpp.gpu": "GPUOpenCL", "lcpp.htp": "HTP0"}[backend_key]
    is_htp = (backend_key == "lcpp.htp")

    # D2 fix: CPU/GPU 단독 실행 시 libggml-hexagon.so 가 NEEDED 로 load 되면서
    # ggml_backend_hexagon_reg() → htpdrv_init() → session create → unsigned PD error abort.
    # LD_PRELOAD stub 으로 ggml_backend_hexagon_reg() + htpdrv_init() 을 override 하여 차단.
    # HTP 는 device_init_failed ⚠ 셀이므로 stub 불필요.
    stub_env: dict = (
        {"LD_PRELOAD": f"{DEVICE_BIN_DIR}/{LCPP_STUB_SO}"}
        if not is_htp
        else {}
    )

    for dtype in ("f16", "q4_0"):
        for op in ALL_OPS:
            if dtype == "q4_0" and op in ACTIVATION_ONLY_OPS:
                if op != "GET_ROWS":
                    if op == "MUL_MAT":
                        for sid in MULMAT_SHAPES:
                            cells.append(_dash_cell(backend_key, op, dtype, sid))
                    else:
                        cells.append(_dash_cell(backend_key, op, dtype, op))
                    continue

            if is_htp:
                # P2: device_init_failed — 모든 cell ⚠
                status = "⚠"
                proxy_note = "device_init_failed"
            elif op in _LCPP_NO_PERF_CASE:
                status = "⚠"
                proxy_note = "no_perf_case"
            elif dtype == "f16" and op in _LCPP_F16_F32_FALLBACK:
                status = "+"
                proxy_note = "dtype_f32_fallback"
            else:
                status = "+"
                proxy_note = ""

            if op == "MUL_MAT":
                for sid, sdef in MULMAT_SHAPES.items():
                    cells.append(FullCell(
                        cell_id=_cell_id(backend_key, op, dtype, sid),
                        backend=backend_key,
                        op=op, dtype=dtype, shape_id=sid,
                        shape_desc=f"{sdef['desc']} ({_LCPP_MULMAT_SHAPE_NOTE})",
                        status=status,
                        proxy_note=_LCPP_MULMAT_SHAPE_NOTE,
                        bin_name="test-backend-ops",
                        bin_dir=DEVICE_BIN_DIR,
                        args=["perf", "-o", "MUL_MAT", "-b", flag,
                              "-p", f"n=1,k=14336"],
                        env=stub_env,
                        latency_pattern=r"([\d.]+)\s*us/run",
                        latency_unit="us",
                        tol_max_abs=5e-2 if dtype == "q4_0" else 1e-2,
                    ))
            else:
                lcpp_op = _lcpp_op_name(op)
                cells.append(FullCell(
                    cell_id=_cell_id(backend_key, op, dtype, op),
                    backend=backend_key,
                    op=op, dtype=dtype, shape_id=op,
                    shape_desc=_op_shape_desc(op),
                    status=status,
                    proxy_note=proxy_note,
                    bin_name="test-backend-ops",
                    bin_dir=DEVICE_BIN_DIR,
                    args=["perf", "-o", lcpp_op, "-b", flag],
                    env=stub_env,
                    latency_pattern=r"([\d.]+)\s*us/run",
                    latency_unit="us",
                    tol_max_abs=1e-2,
                ))
    return cells


def _op_shape_desc(op: str) -> str:
    """op별 Qwen 2.5-1.5B 단일 shape 설명."""
    m = {
        "RMS_NORM":       "[1, 1536] eps=1e-6",
        "ROPE":           "head_dim=128 theta=1e6 (Qwen2 normal)",
        "FLASH_ATTN_EXT": "Q[12,1,128] K/V[2,1024,128] GQA ctx=1024",
        "GET_ROWS":       "embed [151936, 1536] lookup 1 token",
        "SILU":           "[1, 8960] FFN activation",
        "MUL":            "[1, 8960] gate·up elementwise",
        "ADD":            "[1, 1536] residual",
        "SOFT_MAX":       "[12, 1, 1024] FlashAttn fallback ctx",
        "SCALE":          "[12, 1, 1024] Q × 1/sqrt(d_k)",
        "CPY":            "F32→F16 [2, 1, 128] KV store",
        "SET_ROWS":       "[2,1,128] → cache[1024,2,128] KV scatter",
    }
    return m.get(op, op)


def _lcpp_op_name(op: str) -> str:
    """op enum → test-backend-ops -o 인자 매핑."""
    m = {
        "SILU": "UNARY",
        "FLASH_ATTN_EXT": "FLASH_ATTN",
        "SET_ROWS": "SET_ROWS",
    }
    return m.get(op, op)


def build_full_matrix_cells(
    backends_filter: Optional[List[str]] = None,
    ops_filter: Optional[List[str]] = None,
    dtypes_filter: Optional[List[str]] = None,
    cells_filter: Optional[List[str]] = None,
) -> List[FullCell]:
    """196 cell 전체 목록 생성."""
    all_cells: List[FullCell] = []
    all_cells += _ours_cpu_cells()
    all_cells += _ours_gpu_cells()
    all_cells += _ours_htp_cells()
    all_cells += _et_htp_cells()
    all_cells += _lcpp_cells("lcpp.cpu")
    all_cells += _lcpp_cells("lcpp.gpu")
    all_cells += _lcpp_cells("lcpp.htp")

    # 필터 적용
    if cells_filter:
        ids = set(cells_filter)
        all_cells = [c for c in all_cells if c.cell_id in ids]
    if backends_filter:
        bs = set(backends_filter)
        all_cells = [c for c in all_cells if c.backend in bs]
    if ops_filter:
        os_set = set(ops_filter)
        all_cells = [c for c in all_cells if c.op in os_set]
    if dtypes_filter:
        ds = set(dtypes_filter)
        all_cells = [c for c in all_cells if c.dtype in ds]

    return all_cells


# ==============================================================================
# Output parsing (공통)
# ==============================================================================

@dataclass
class TrialResult:
    cell_id: str
    round_idx: int
    is_warmup: bool
    ok: bool
    latency_ms: Optional[float] = None  # 항상 ms 단위로 정규화
    latency_unit_raw: str = "ms"        # 원본 단위
    latency_raw: Optional[float] = None  # 원본 단위 값
    max_abs_err: Optional[float] = None
    cosine: Optional[float] = None
    status_note: str = ""  # proxy_note / no_perf_case / device_init_failed 등
    stderr: str = ""
    raw_stdout: str = ""
    thermal_before: Dict[str, float] = field(default_factory=dict)
    thermal_after: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


_LAT_PATTERNS = [
    # JSON structured output (ms 직접 반환)
    ("json_latency_ms", lambda s: _parse_json_field(s, "latency_ms")),
    ("json_median_ns",  lambda s: _ns_to_ms(_parse_json_field(s, "median_ns"))),
    ("json_median_ms",  lambda s: _parse_json_field(s, "median_ms")),
    # ms 단위 평문 패턴
    ("latency_re", lambda s: _parse_regex(s, r"latency[:\s]+([\d.]+)\s*ms")),
    ("median_re",  lambda s: _parse_regex(s, r"median[:\s]+([\d.]+)\s*ms")),
    ("tbt_re",     lambda s: _parse_regex(s, r"tbt[:\s]+([\d.]+)\s*ms")),
    # us 단위 평문 패턴 → ms 변환 (D3 fix: ours.htp F16 bins 출력 형식)
    # 형식: "mean=97.27 us" / "HTP F16: mean=211.80 us" / "97.27 us/op"
    # last-match: stdout에 CPU baseline(앞) + HTP(뒤) 순서이므로 마지막 값이 NPU 결과
    ("mean_us_re",   lambda s: _us_to_ms(_parse_regex_last(s, r"mean=([\d.]+)\s*(?:us|μs|µs)"))),
    ("median_us_re", lambda s: _us_to_ms(_parse_regex_last(s, r"median=([\d.]+)\s*(?:us|μs|µs)"))),
    ("xus_op_re",    lambda s: _us_to_ms(_parse_regex_last(s, r"([\d.]+)\s*(?:us|μs|µs)/op"))),
    ("xus_run_re",   lambda s: _us_to_ms(_parse_regex_last(s, r"([\d.]+)\s*(?:us|μs|µs)/run"))),
    # ns 단위 평문 패턴 → ms 변환 (방어적)
    ("mean_ns_re",   lambda s: _ns_to_ms(_parse_regex_last(s, r"mean=([\d.]+)\s*ns"))),
    ("median_ns_re", lambda s: _ns_to_ms(_parse_regex_last(s, r"median=([\d.]+)\s*ns"))),
    # 최후 fallback (ms 단위 숫자)
    ("any_ms_re",  lambda s: _parse_regex(s, r"([\d.]+)\s*ms")),
]


def _ns_to_ms(v: Optional[float]) -> Optional[float]:
    return v / 1e6 if v is not None else None


def _us_to_ms(v: Optional[float]) -> Optional[float]:
    return v / 1e3 if v is not None else None


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
    m = re.search(pat, s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _parse_regex_last(s: str, pat: str) -> Optional[float]:
    """re.findall로 모든 매칭을 찾고 마지막 값을 반환.

    stdout에 CPU baseline + HTP 결과가 순서대로 나올 때
    마지막 값이 HTP(NPU) 결과이므로 last-match가 정확하다.
    """
    matches = re.findall(pat, s, flags=re.IGNORECASE)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
            return None
    return None


def parse_trial_output(
    stdout: str,
    custom_pattern: Optional[str] = None,
    latency_unit: str = "ms",
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (latency_ms, max_abs_err, cosine).

    custom_pattern hit → lat_raw은 raw 값(latency_unit 단위)이므로 변환 적용.
    _LAT_PATTERNS fallback hit → 이미 ms로 정규화된 값이므로 변환 skip.
    """
    lat_ms = None

    if custom_pattern:
        lat_raw = _parse_regex(stdout, custom_pattern)
        if lat_raw is not None:
            # custom_pattern은 cell.latency_unit 단위 → ms 변환
            if latency_unit == "ns":
                lat_ms = lat_raw / 1e6
            elif latency_unit == "us":
                lat_ms = lat_raw / 1e3
            else:
                lat_ms = lat_raw

    if lat_ms is None:
        # _LAT_PATTERNS는 모두 ms 단위로 정규화된 값을 반환 → 변환 불필요
        for _, fn in _LAT_PATTERNS:
            v = fn(stdout)
            if v is not None:
                lat_ms = v
                break

    err = _parse_json_field(stdout, "max_abs_err")
    if err is None:
        # D4 fix: '=' 도 구분자로 허용 (stdout 형식: max_abs_err=3.537e-2)
        err = _parse_regex(stdout, r"max[_\s]abs[_\s]err[=:\s]+([\d.eE+-]+)")
    cos = _parse_json_field(stdout, "cosine")
    if cos is None:
        cos = _parse_regex(stdout, r"cosine[:\s]+([\d.eE+-]+)")
    return lat_ms, err, cos


# ==============================================================================
# Outlier rejection + stats (공통)
# ==============================================================================

def tukey_iqr(values: List[float], k: float = 1.5) -> Tuple[List[float], List[float]]:
    if len(values) < 4:
        return values[:], []
    q1, q3 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    filtered = [v for v in values if lo <= v <= hi]
    outliers = [v for v in values if v < lo or v > hi]
    return filtered, outliers


def cell_stats(trials: List[TrialResult]) -> Dict:
    measure = [t for t in trials if (not t.is_warmup) and t.ok and t.latency_ms is not None]
    if not measure:
        return {"n": 0, "n_valid": 0, "median_ms": None, "cv_pct": None}
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


# ==============================================================================
# Cooldown / thermal control (공통)
# ==============================================================================

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


# ==============================================================================
# Trial runner (legacy + full matrix 공용)
# ==============================================================================

def run_trial_legacy(
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
                cell_id=cell.name, round_idx=round_idx, is_warmup=is_warmup, ok=False,
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
        stdout = p.stdout
        stderr = p.stderr
    except subprocess.TimeoutExpired:
        return TrialResult(
            cell_id=cell.name, round_idx=round_idx, is_warmup=is_warmup, ok=False,
            error="timeout after 180s",
            thermal_before=thermal_before,
        )

    thermal_after = adb.read_zones()
    combined = (stdout or "") + "\n" + (stderr or "")
    lat_ms, err, cos = parse_trial_output(combined, cell.latency_pattern)
    parsed_latency = lat_ms is not None
    if lat_ms is None:
        lat_ms = elapsed_wall

    accuracy_ok = True
    if err is not None and err > cell.tolerance_max_abs:
        accuracy_ok = False
    if cos is not None and cos < cell.tolerance_cosine:
        accuracy_ok = False

    valid = parsed_latency and accuracy_ok

    return TrialResult(
        cell_id=cell.name, round_idx=round_idx, is_warmup=is_warmup,
        ok=valid,
        latency_ms=lat_ms, max_abs_err=err, cosine=cos,
        stderr=stderr[-400:] if stderr else "",
        raw_stdout=stdout[-1200:] if stdout else "",
        thermal_before=thermal_before, thermal_after=thermal_after,
        error=None if valid else (
            "binary crashed (no latency in stdout)" if not parsed_latency
            else "accuracy out of tolerance"
        ),
    )


def run_trial_full(
    adb: Adb,
    cell: FullCell,
    round_idx: int,
    is_warmup: bool,
    log_fn,
    dry_run: bool = False,
) -> TrialResult:
    """full matrix 단일 trial 실행."""
    # ⚠ / ✗ / — cell: 즉시 반환 (측정 안 함)
    if not cell.is_measurable():
        return TrialResult(
            cell_id=cell.cell_id, round_idx=round_idx, is_warmup=is_warmup,
            ok=True,
            latency_ms=None,
            status_note=cell.proxy_note or cell.status,
            thermal_before={}, thermal_after={},
        )

    thermal_before = {}
    if not dry_run:
        thermal_before = adb.read_zones()
        max_before = max(thermal_before.values())
        if max_before >= TRIGGER_TEMP_C:
            log_fn(f"  [trigger] max={max_before:.1f}°C ≥ {TRIGGER_TEMP_C}°C — extra cooldown")
            if not wait_thermal_below(adb, RECOVERY_TEMP_C, COOLDOWN_TRIGGER_S, "trigger-recovery", log_fn):
                return TrialResult(
                    cell_id=cell.cell_id, round_idx=round_idx, is_warmup=is_warmup, ok=False,
                    error=f"thermal trigger not recovered (>{RECOVERY_TEMP_C}°C)",
                    thermal_before=thermal_before,
                )
            thermal_before = adb.read_zones()

    if dry_run:
        log_fn(f"  [dry-run] SKIP execution: {cell.cell_id}")
        return TrialResult(
            cell_id=cell.cell_id, round_idx=round_idx, is_warmup=is_warmup,
            ok=True, latency_ms=None,
            status_note="dry_run_skip",
            thermal_before={}, thermal_after={},
        )

    cmd = cell.adb_cmd()
    try:
        t0 = time.time()
        p = subprocess.run(
            ["adb", "-s", adb.serial, "shell", cmd],
            capture_output=True, text=True, timeout=300,
        )
        elapsed_wall_ms = (time.time() - t0) * 1000.0
        stdout = p.stdout or ""
        stderr = p.stderr or ""
    except subprocess.TimeoutExpired:
        return TrialResult(
            cell_id=cell.cell_id, round_idx=round_idx, is_warmup=is_warmup, ok=False,
            error="timeout after 300s",
            thermal_before=thermal_before,
        )

    thermal_after = adb.read_zones()
    combined = stdout + "\n" + stderr
    lat_ms, err, cos = parse_trial_output(combined, cell.latency_pattern, cell.latency_unit)
    parsed = lat_ms is not None
    if not parsed:
        lat_ms = elapsed_wall_ms

    accuracy_ok = True
    if err is not None and err > cell.tol_max_abs:
        accuracy_ok = False
    if cos is not None and cos < cell.tol_cosine:
        accuracy_ok = False

    valid = parsed and accuracy_ok

    return TrialResult(
        cell_id=cell.cell_id, round_idx=round_idx, is_warmup=is_warmup,
        ok=valid,
        latency_ms=lat_ms,
        latency_unit_raw=cell.latency_unit,
        max_abs_err=err, cosine=cos,
        status_note=cell.proxy_note,
        stderr=stderr[-400:],
        raw_stdout=stdout[-1200:],
        thermal_before=thermal_before, thermal_after=thermal_after,
        error=None if valid else (
            "no latency parsed" if not parsed else "accuracy out of tolerance"
        ),
    )


# ==============================================================================
# Matrix orchestration (full mode)
# ==============================================================================

def run_full_matrix(
    adb: Adb,
    cells: List[FullCell],
    out_dir: Path,
    rounds: int,
    seed: int,
    log_fn,
    dry_run: bool = False,
) -> Dict[str, List[TrialResult]]:
    measurable = [c for c in cells if c.is_measurable()]
    skipped = [c for c in cells if not c.is_measurable()]

    log_fn(f"[full-matrix] {len(cells)} total cells")
    log_fn(f"[full-matrix] measurable: {len(measurable)}")
    log_fn(f"[full-matrix] skipped (✗/⚠/—): {len(skipped)}")
    if dry_run:
        log_fn(f"[full-matrix] DRY-RUN MODE — printing cell definitions, no execution")
        _print_cell_definitions(cells, log_fn)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    thermal_log_path = out_dir / "thermal_log.csv"
    seq_path = out_dir / "round_sequence.json"

    with thermal_log_path.open("w") as tf:
        tf.write("timestamp,phase,cell_id,round_idx," + ",".join(ZONES.keys()) + "\n")

    rng = random.Random(seed)
    sequence: List[List[str]] = []
    results: Dict[str, List[TrialResult]] = {c.cell_id: [] for c in cells}

    if not dry_run:
        log_fn(f"[matrix] session warmup {SESSION_WARMUP_S}s")
        time.sleep(SESSION_WARMUP_S)

    for r in range(rounds):
        # round-robin shuffle: measurable cell만 shuffle, skip cell은 라운드 내 1회 처리
        order_ids = [c.cell_id for c in measurable]
        rng.shuffle(order_ids)
        # skip cell을 맨 뒤에 붙여 1-round에 1회 처리
        skip_ids = [c.cell_id for c in skipped]
        full_order = order_ids + skip_ids
        sequence.append(full_order)
        is_warmup = (r < WARMUP_TRIALS)
        log_fn(f"\n=== round {r+1}/{rounds} ({'WARMUP' if is_warmup else 'MEASURE'}) ===")

        for cid in full_order:
            cell = next(c for c in cells if c.cell_id == cid)

            # inter-cell cooldown (measurable cell만, 첫 번째 cell 제외)
            if cell.is_measurable() and results[cid] and not dry_run:
                cooldown(adb, COOLDOWN_MIN_S, COOLDOWN_MAX_S, f"inter-cell {cid}", log_fn)

            log_fn(f"\n--- {cid} r{r+1} ({'wup' if is_warmup else 'meas'}) status={cell.status} ---")
            tr = run_trial_full(adb, cell, r, is_warmup, log_fn, dry_run=dry_run)
            results[cid].append(tr)

            # raw JSON (skip cell도 status 기록)
            tr_dict = asdict(tr)
            with (raw_dir / f"{cid}_round{r:02d}.json").open("w") as f:
                json.dump(tr_dict, f, indent=2)

            # thermal log (measurable only)
            if cell.is_measurable() and not dry_run:
                with thermal_log_path.open("a") as tf:
                    ts = int(time.time())
                    bb = ",".join(f"{tr.thermal_before.get(k, 0):.2f}" for k in ZONES.keys())
                    aa = ",".join(f"{tr.thermal_after.get(k, 0):.2f}" for k in ZONES.keys())
                    tf.write(f"{ts},before,{cid},{r},{bb}\n")
                    tf.write(f"{ts},after,{cid},{r},{aa}\n")

            lat_str = f"{tr.latency_ms:.3f}ms" if tr.latency_ms is not None else "N/A"
            log_fn(f"  ok={tr.ok} lat={lat_str} note={tr.status_note!r}")

            if cell.is_measurable() and not dry_run:
                time.sleep(INTER_TRIAL_S)

        # skip cell은 r=0 이후 한 번만 처리 (이미 results에 있으니 skip)
        if r == 0:
            # 첫 round에 skip cell 한 번 처리 완료
            pass

        # inter-round cooldown
        if r < rounds - 1 and not dry_run:
            cooldown(adb, INTER_ROUND_S, INTER_ROUND_MAX_S, f"inter-round {r+1}->{r+2}", log_fn)

    with seq_path.open("w") as f:
        json.dump(sequence, f, indent=2)

    return results


def _print_cell_definitions(cells: List[FullCell], log_fn) -> None:
    """dry-run: cell 정의 출력."""
    status_counts: Dict[str, int] = {}
    log_fn(f"\n{'cell_id':<55} {'backend':<10} {'op':<18} {'dtype':<6} {'status':<4} {'proxy_note'}")
    log_fn("-" * 130)
    for c in sorted(cells, key=lambda x: (x.backend, x.op, x.dtype, x.shape_id)):
        log_fn(f"{c.cell_id:<55} {c.backend:<10} {c.op:<18} {c.dtype:<6} {c.status:<4} {c.proxy_note}")
        status_counts[c.status] = status_counts.get(c.status, 0) + 1
    log_fn(f"\n[dry-run] status breakdown: {status_counts}")
    log_fn(f"[dry-run] total: {len(cells)}")
    measurable = sum(1 for c in cells if c.is_measurable())
    log_fn(f"[dry-run] measurable (✓/+): {measurable}")


# ==============================================================================
# Aggregation + report (full mode)
# ==============================================================================

def aggregate_full(
    results: Dict[str, List[TrialResult]],
    cells: List[FullCell],
    out_dir: Path,
) -> None:
    csv_path = out_dir / "aggregated.csv"
    summary_path = out_dir / "summary.json"

    summary = {}
    with csv_path.open("w") as f:
        f.write(
            "cell_id,backend,op,dtype,shape_id,status,"
            "n_raw,n_outlier,n_valid,median_ms,stdev_ms,cv_pct,"
            "p25_ms,p75_ms,min_ms,max_ms,max_abs_err_max,cosine_min,"
            "proxy_note\n"
        )
        for c in cells:
            st = cell_stats(results.get(c.cell_id, []))
            summary[c.cell_id] = {
                "meta": {
                    "backend": c.backend, "op": c.op, "dtype": c.dtype,
                    "shape_id": c.shape_id, "status": c.status,
                    "proxy_note": c.proxy_note,
                },
                **st,
            }
            f.write(",".join(str(x) for x in [
                c.cell_id, c.backend, c.op, c.dtype, c.shape_id, c.status,
                st.get("n_raw", 0), st.get("n_outlier", 0), st.get("n_valid", 0),
                st.get("median_ms"), st.get("stdev_ms"), st.get("cv_pct"),
                st.get("p25_ms"), st.get("p75_ms"),
                st.get("min_ms"), st.get("max_ms"),
                st.get("max_abs_err_max"), st.get("cosine_min"),
                c.proxy_note,
            ]) + "\n")

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


def generate_report_full(
    results: Dict[str, List[TrialResult]],
    cells: List[FullCell],
    out_dir: Path,
) -> None:
    md = out_dir / "report.md"
    lines = [
        "# Qwen 2.5-1.5B Microbench Full Matrix Report",
        "",
        f"- 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Trial protocol: warmup {WARMUP_TRIALS} + measure {MEASURE_TRIALS}",
        f"- device: Galaxy S25 (SM8750, Hexagon V79, Adreno 830)",
        f"- model: Qwen 2.5-1.5B (hidden=1536, ffn=8960, vocab=151936, n_kv=2)",
        "",
        "## Status 범례",
        "",
        "| symbol | 의미 |",
        "|---|---|",
        "| ✓  | existing (μ-Q1 inherit) |",
        "| +  | new bin (측정 대상) |",
        "| ⚠  | perf abort / device_init_failed / no_perf_case |",
        "| ✗  | hard 미지원 |",
        "| —  | fair 비교 부적합 (activation-only Q4_0 row) |",
        "| proxy=note | 대리 측정 패턴 (P1a 발견) |",
        "| dtype_f32_fallback | F16 요청 시 F32 케이스 사용 (P2 발견) |",
        "| shape_mismatch=note | L.cpp 고정 shape 사용 (Qwen shape 없음) |",
        "| no_perf_case | test-backend-ops perf case 없는 op |",
        "| device_init_failed | L.cpp HTP0 session create 실패 (P2) |",
        "",
    ]

    # 7-column fair-pair 표 (op 단위)
    lines += [
        "## 7-Backend Fair-Pair 결과 매트릭스",
        "",
        "단위: ms/call (median, warmup 3 + measure 10, Tukey 1.5×IQR)",
        "",
    ]

    BACKEND_ORDER = ["ours.cpu", "ours.gpu", "ours.htp", "et.htp", "lcpp.cpu", "lcpp.gpu", "lcpp.htp"]

    for op in ALL_OPS:
        lines.append(f"### {op}")
        lines.append("")
        header = "| shape | dtype | " + " | ".join(BACKEND_ORDER) + " |"
        sep = "|---|---|" + "---|" * len(BACKEND_ORDER)
        lines.append(header)
        lines.append(sep)

        shapes = list(MULMAT_SHAPES.keys()) if op == "MUL_MAT" else [op]
        for sid in shapes:
            for dtype in ("f16", "q4_0"):
                row_parts = [sid, dtype]
                for bk in BACKEND_ORDER:
                    cid = _cell_id(bk, op, dtype, sid)
                    cell = next((c for c in cells if c.cell_id == cid), None)
                    if cell is None or cell.status == "—":
                        row_parts.append("—")
                        continue
                    if not cell.is_measurable():
                        row_parts.append(f"{cell.status}")
                        continue
                    st = cell_stats(results.get(cid, []))
                    if st.get("n_valid", 0) == 0:
                        row_parts.append("N/A")
                    else:
                        v = st.get("median_ms")
                        cv = st.get("cv_pct")
                        row_parts.append(f"{v:.3f}" if v else "-")
                lines.append("| " + " | ".join(row_parts) + " |")
        lines.append("")

    # P1a proxy 발견 요약
    proxy_cells = [c for c in cells if c.proxy_note]
    if proxy_cells:
        lines += [
            "## P1a Proxy 측정 패턴",
            "",
            "| cell_id | proxy_note |",
            "|---|---|",
        ]
        for c in proxy_cells:
            lines.append(f"| {c.cell_id} | {c.proxy_note} |")
        lines.append("")

    # P2 발견 요약
    lines += [
        "## P2 L.cpp 제약 사항",
        "",
        "- **lcpp.htp 19 cell**: `⚠ device_init_failed` (unsigned PD 부재, SM8750 S25)",
        "  - `ggml-hex: FastRPC capability query failed`, `failed to enable unsigned PD`",
        "- **perf case 없는 op 7개**: SILU/MUL/RMS_NORM/GET_ROWS/SCALE/SET_ROWS/FLASH_ATTN_EXT",
        "  - `⚠ no_perf_case` 마킹, support/test 모드만 가능",
        "- **F16 dtype 없는 op 4개**: ADD/ROPE/SOFT_MAX/CPY",
        "  - F32 케이스만 존재 → `dtype_f32_fallback` 마킹",
        "- **MUL_MAT shape mismatch**: test-backend-ops 고정 shape m=4096,k=14336 사용",
        "  - Qwen 2.5-1.5B 정확 shape (m=8960/151936/2048, k=1536) 없음",
        "",
        "## P1d Ours-NPU F16 비대칭 지원 (DSP-side 분석)",
        "",
        "| op | F16 support | 결정 |",
        "|---|---|---|",
        "| MUL_MAT | ✓ matmul-ops.c f16-f32 path | + new bin |",
        "| ADD | ✓ binary-ops.c hvx_add_f16_* | + new bin |",
        "| MUL | ✓ binary-ops.c hvx_mul_f16_* | + new bin |",
        "| FLASH_ATTN_EXT | ✓ flash-attn-ops.c:618 attempt | + new bin (GREEN/✗ 확정 필요) |",
        "| SILU | ✗ unary-ops.c F32 only (NO_SUPPORT) | ✗ (evidence bin 작성) |",
        "| RMS_NORM / ROPE / GET_ROWS / SOFT_MAX | ✗ F32 only | ✗ (SILU evidence 동일 인용) |",
        "| SCALE / CPY / SET_ROWS | ✗ init_*_req 미구현 | ✗ |",
        "",
        "## P1c ExecuTorch 제약 사항",
        "",
        "- **FLASH_ATTN_EXT**: Dynamo FX export 실패 (✗) — F.scaled_dot_product_attention 미지원",
        "- **CPY**: aten._to_copy delegate 실패 (✗)",
        "- **GET_ROWS mm_lmh**: vocab=151936 OOM → paper annotation 필요 (vocab=1024 fallback)",
        "",
    ]

    md.write_text("\n".join(lines))


# ==============================================================================
# Dependency verification (dry-run 시 호출)
# ==============================================================================

def verify_dependencies(adb: Adb, cells: List[FullCell], log_fn) -> bool:
    """device 위 bin 존재 여부 확인 (dry-run / preflight)."""
    ok = True
    checked: Dict[str, bool] = {}

    for c in cells:
        if not c.is_measurable():
            continue
        if not c.bin_name:
            continue
        bin_path = f"{c.bin_dir}/{c.bin_name}"
        if bin_path in checked:
            continue
        exists = adb.file_exists(bin_path)
        checked[bin_path] = exists
        mark = "OK" if exists else "MISSING"
        log_fn(f"  [{mark}] {bin_path}  ({c.backend})")
        if not exists:
            ok = False

    return ok


# ==============================================================================
# Legacy matrix orchestration (backward-compat)
# ==============================================================================

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

            if results[cname]:
                cooldown(adb, COOLDOWN_MIN_S, COOLDOWN_MAX_S, f"inter-cell {cname}", log_fn)

            log_fn(f"\n--- {cname} round {r+1} ({'warmup' if is_warmup else 'measure'}) ---")
            tr = run_trial_legacy(adb, cell, r, is_warmup, work_dir, log_fn)
            results[cname].append(tr)

            tr_dict = asdict(tr)
            with (raw_dir / f"{cname}_round{r:02d}.json").open("w") as f:
                json.dump(tr_dict, f, indent=2)

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

        if r < rounds - 1:
            cooldown(adb, INTER_ROUND_S, INTER_ROUND_MAX_S, f"inter-round {r+1}->{r+2}", log_fn)

    with seq_path.open("w") as f:
        json.dump(sequence, f, indent=2)

    return results


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


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="QNN microbench matrix driver (legacy 9-cell + full 196-cell)"
    )
    parser.add_argument("--serial", default="R3CY408S5SB",
                        help="ADB device serial (default: R3CY408S5SB = S25)")
    parser.add_argument("--matrix", choices=["legacy", "full"], default="legacy",
                        help="legacy: 기존 9-cell (backward-compat) | full: 196 cell P3 확장")
    parser.add_argument("--out", "--output-dir", dest="out",
                        help="Output directory (full mode에서는 --output-dir 사용 가능)")
    # Full matrix 전용 옵션
    parser.add_argument("--backends", default="",
                        help="[full] 쉼표 구분 backend 목록 (예: ours.cpu,ours.gpu). 빈값=전체")
    parser.add_argument("--ops", default="",
                        help="[full] 쉼표 구분 op 목록 (예: MUL_MAT,RMS_NORM). 빈값=전체")
    parser.add_argument("--dtypes", default="",
                        help="[full] 쉼표 구분 dtype 목록 (f16,q4_0). 빈값=전체")
    parser.add_argument("--cells", default="",
                        help="[legacy] 쉼표 구분 cell 이름 (M3,M4 등). 빈값=기본 활성. "
                             "[full] cell_id 직접 지정 (예: ours_cpu_MUL_MAT_f16_mm_ffn)")
    # 공통 옵션
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help=f"Total rounds (default {ROUNDS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Cell 정의 출력 후 측정 안 함 (full mode에서는 196 cell 전체 정의 출력)")
    parser.add_argument("--dry-run-setup", action="store_true",
                        help="[legacy] ADB + zones + zombie check만")
    parser.add_argument("--quick", action="store_true",
                        help="단축 cooldown (inter-cell 10s, inter-round 30s) — protocol 검증용")
    parser.add_argument("--verify-deps", action="store_true",
                        help="[full] device 위 bin 존재 여부 확인 후 종료")
    args = parser.parse_args()

    if args.quick:
        global COOLDOWN_MIN_S, COOLDOWN_MAX_S, INTER_ROUND_S, INTER_ROUND_MAX_S, SESSION_WARMUP_S
        COOLDOWN_MIN_S = 10
        COOLDOWN_MAX_S = 60
        INTER_ROUND_S = 30
        INTER_ROUND_MAX_S = 90
        SESSION_WARMUP_S = 15

    # --out 기본값 설정
    if not args.out:
        if args.matrix == "full":
            args.out = str(REPO_ROOT / "papers/eurosys2027/_workspace/experiment/microbench_full_matrix_2026_05_28")
        else:
            parser.error("--out is required")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "driver.log"
    log_f = log_path.open("w")
    def log_fn(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log_fn(f"[init] matrix={args.matrix} device={args.serial}")
    log_fn(f"[init] out={out_dir}")
    log_fn(f"[init] dry_run={args.dry_run}")

    adb = Adb(args.serial)

    # ==== FULL MATRIX mode ====
    if args.matrix == "full":
        backends_filter = [b.strip() for b in args.backends.split(",") if b.strip()] or None
        ops_filter = [o.strip() for o in args.ops.split(",") if o.strip()] or None
        dtypes_filter = [d.strip() for d in args.dtypes.split(",") if d.strip()] or None
        cells_filter = [c.strip() for c in args.cells.split(",") if c.strip()] or None

        cells = build_full_matrix_cells(backends_filter, ops_filter, dtypes_filter, cells_filter)
        log_fn(f"[full] total cells: {len(cells)}")

        if args.dry_run:
            _print_cell_definitions(cells, log_fn)
            log_fn(f"\n[dry-run] dependency check...")
            verify_dependencies(adb, cells, log_fn)
            log_f.close()
            return 0

        if args.verify_deps:
            ok = verify_dependencies(adb, cells, log_fn)
            log_f.close()
            return 0 if ok else 1

        # Preflight
        zombies = adb.zombie_check()
        if zombies:
            log_fn(f"[preflight] FAIL — zombies:")
            for z in zombies:
                log_fn(f"  {z}")
            log_f.close()
            return 3

        name, t = adb.max_zone_temp()
        log_fn(f"[preflight] max_zone={name}={t:.1f}°C")
        if t > START_TEMP_C:
            if not wait_thermal_below(adb, START_TEMP_C, 300, "preflight", log_fn):
                log_fn("[preflight] cooldown timeout — abort")
                log_f.close()
                return 4

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
            "total_cells": len(cells),
            "measurable_cells": sum(1 for c in cells if c.is_measurable()),
            "backends_filter": backends_filter,
            "ops_filter": ops_filter,
            "dtypes_filter": dtypes_filter,
        }
        (out_dir / "env.json").write_text(json.dumps(env, indent=2))

        try:
            results = run_full_matrix(adb, cells, out_dir, args.rounds, args.seed, log_fn)
        except KeyboardInterrupt:
            log_fn("[interrupted]")
            log_f.close()
            return 130
        except Exception as e:
            log_fn(f"[error] {type(e).__name__}: {e}")
            log_f.close()
            raise

        aggregate_full(results, cells, out_dir)
        generate_report_full(results, cells, out_dir)

        log_fn(f"\n[done] outputs:")
        log_fn(f"  {out_dir}/raw/")
        log_fn(f"  {out_dir}/aggregated.csv")
        log_fn(f"  {out_dir}/summary.json")
        log_fn(f"  {out_dir}/thermal_log.csv")
        log_fn(f"  {out_dir}/report.md")
        log_fn(f"  {out_dir}/env.json")
        log_f.close()
        return 0

    # ==== LEGACY mode ====
    enable = [s.strip() for s in args.cells.split(",") if s.strip()]
    legacy_cells = build_cells(enable)

    log_fn(f"[init] enabled cells: {[c.name for c in legacy_cells if c.enabled]}")

    zombies = adb.zombie_check()
    if zombies:
        log_fn(f"[preflight] FAIL — zombies:")
        for z in zombies:
            log_fn(f"  {z}")
        log_f.close()
        return 3

    name, t = adb.max_zone_temp()
    log_fn(f"[preflight] max_zone={name}={t:.1f}°C")
    if t > START_TEMP_C:
        if not wait_thermal_below(adb, START_TEMP_C, 300, "preflight", log_fn):
            log_fn("[preflight] cooldown timeout — abort")
            log_f.close()
            return 4

    airplane = adb.shell("settings get global airplane_mode_on", check=False)
    log_fn(f"[preflight] airplane_mode_on={airplane!r}")
    if airplane.strip() != "1":
        log_fn("[preflight] WARN: airplane mode is OFF")

    if args.dry_run_setup:
        log_fn("[dry-run-setup] preflight OK, exiting")
        log_f.close()
        return 0

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

    try:
        results = run_matrix(adb, legacy_cells, out_dir, args.rounds, args.seed, log_fn)
    except KeyboardInterrupt:
        log_fn("[interrupted]")
        log_f.close()
        return 130
    except Exception as e:
        log_fn(f"[error] {type(e).__name__}: {e}")
        log_f.close()
        raise

    aggregate(results, legacy_cells, out_dir)
    generate_report(results, legacy_cells, out_dir)

    log_fn(f"\n[done] outputs:")
    log_fn(f"  {out_dir}/raw/")
    log_fn(f"  {out_dir}/aggregated.csv")
    log_fn(f"  {out_dir}/summary.json")
    log_fn(f"  {out_dir}/thermal_log.csv")
    log_fn(f"  {out_dir}/report.md")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
