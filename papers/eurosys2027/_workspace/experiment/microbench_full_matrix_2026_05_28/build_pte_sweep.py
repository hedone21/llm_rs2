#!/usr/bin/env python3
"""build_pte_sweep.py — Qwen 2.5-1.5B full-matrix PTE sweep builder.

12 op × {use_fp16, use_8a4w} = 24 .pte + MUL_MAT 추가 shape 4 = 총 28 .pte.

사용법:
  # dry-run (venv 가용성 + 1-cell test):
  python build_pte_sweep.py --dry-run

  # 전체 sweep:
  python build_pte_sweep.py

  # 특정 op × dtype:
  python build_pte_sweep.py --op MUL_MAT --shape mm_ffn --dtype use_fp16

모델 파라미터 (Qwen 2.5-1.5B):
  hidden=1536, n_heads=12, n_kv=2 (GQA 6:1), head_dim=128, FFN=8960,
  vocab=151936, n_layers=28, max_ctx=1024 (representative)

MUL_MAT 3 shape:
  mm_ffn  : K=1536, N=8960  (FFN gate = up, μ-Q1 inherit)
  mm_lmh  : K=1536, N=151936 (LM head)
  mm_qkv  : K=1536, N=2048  (Q=1536 + K=256 + V=256 fused)

ExecuTorch QNN HTP backend:
  use_fp16 → generate_htp_compiler_spec(use_fp16=True)
  use_8a4w → PT2E W4A8 quantize (QuantDtype.use_8a4w, seed=42)

SoC: SM8750 (Snapdragon 8 Elite, HtpArch.V79)

Unsupported op (✗ 마킹 대상):
  FLASH_ATTN_EXT — ExecuTorch QNN backend 가 F.scaled_dot_product_attention
                    을 native fused op 으로 지원하지 않아 분해됨. ✗ 로 기록.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# ---------------------------------------------------------------------------
# Qwen 2.5-1.5B 모델 파라미터
# ---------------------------------------------------------------------------
HIDDEN = 1536
N_HEADS = 12
N_KV_HEADS = 2
HEAD_DIM = 128
FFN_DIM = 8960
VOCAB = 151936
MAX_CTX = 1024  # representative decode context
RMS_EPS = 1e-6

SOC = QcomChipset.SM8750

# ---------------------------------------------------------------------------
# 셀 정의: (op, shape_id, K, N_or_extra)
# ---------------------------------------------------------------------------
# MUL_MAT 3 shape (p0 §3 결정)
MULMAT_SHAPES = {
    "mm_ffn": (HIDDEN, FFN_DIM),   # FFN gate / up
    "mm_lmh": (HIDDEN, VOCAB),     # LM head
    "mm_qkv": (HIDDEN, 2048),      # Q=1536+K=256+V=256 fused
}

# op 목록 (12개, SWIGLU 제외)
ALL_OPS = [
    "MUL_MAT",
    "RMS_NORM",
    "ROPE",
    "FLASH_ATTN_EXT",
    "GET_ROWS",
    "SILU",
    "MUL",
    "ADD",
    "SOFT_MAX",
    "SCALE",
    "CPY",
    "SET_ROWS",
]

# dtype 목록
ALL_DTYPES = ["use_fp16", "use_8a4w"]

# ---------------------------------------------------------------------------
# nn.Module 정의 (op 별)
# ---------------------------------------------------------------------------


class MatMulModel(nn.Module):
    """단일 Linear (bias=False). K × N shape."""

    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.linear = nn.Linear(k, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RMSNormModel(nn.Module):
    """RMSNorm [1, hidden]. Qwen2 eps=1e-6."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))
        self.eps = RMS_EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, hidden]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RoPEModel(nn.Module):
    """RoPE for Qwen2: head_dim=128, theta=1e6, single-token decode (seq=1).

    입력: q [1, N_HEADS, HEAD_DIM] (q 만 적용; k 는 동일 구조라 대표).
    """

    def __init__(self) -> None:
        super().__init__()
        half = HEAD_DIM // 2
        theta = 1e6
        inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer("inv_freq", inv_freq)  # [half]

    def forward(self, q: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # q: [1, N_HEADS, HEAD_DIM], pos: [1] long
        # freqs: [1, head_dim/2]
        freqs = pos.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [1, half]
        emb = torch.cat([freqs, freqs], dim=-1)  # [1, head_dim]
        cos = emb.cos().unsqueeze(1)  # [1, 1, head_dim]
        sin = emb.sin().unsqueeze(1)
        # rotate_half
        half = q.shape[-1] // 2
        q1, q2 = q[..., :half], q[..., half:]
        q_rot = torch.cat([-q2, q1], dim=-1)
        return q * cos + q_rot * sin


class FlashAttnModel(nn.Module):
    """F.scaled_dot_product_attention (hs=128, nh=12, nkv=2, ctx=MAX_CTX).

    ExecuTorch QNN backend 는 fused FA 미지원 → ✗ 마킹 대상.
    빌드 시도 → 에러 또는 분해된 .pte 생성 여부 기록.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,   # [1, N_HEADS, 1, HEAD_DIM]
        k: torch.Tensor,   # [1, N_KV_HEADS, MAX_CTX, HEAD_DIM]
        v: torch.Tensor,   # [1, N_KV_HEADS, MAX_CTX, HEAD_DIM]
    ) -> torch.Tensor:
        # GQA: expand kv
        k = k.expand(-1, N_HEADS // N_KV_HEADS, -1, -1)
        v = v.expand(-1, N_HEADS // N_KV_HEADS, -1, -1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)


class GetRowsModel(nn.Module):
    """Embedding lookup: embed_table[vocab, hidden], single-token.

    vocab=151936 는 너무 커서 ExecuTorch export 에서 OOM 위험 → 대표 1024 vocab 로 build,
    paper annotation 에 명시. full vocab 는 qnn_executor_runner 실행 시 확인.
    """

    def __init__(self, vocab: int = 1024) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, HIDDEN)
        self.vocab = vocab

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [1] long
        return self.embed(idx)


class SiluModel(nn.Module):
    """SiLU activation [1, FFN_DIM]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class MulModel(nn.Module):
    """Element-wise multiply [1, FFN_DIM]."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class AddModel(nn.Module):
    """Element-wise add [1, HIDDEN]."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class SoftMaxModel(nn.Module):
    """Softmax over last dim [N_HEADS, 1, MAX_CTX]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class ScaleModel(nn.Module):
    """Tensor × scalar (1/sqrt(head_dim))."""

    def __init__(self) -> None:
        super().__init__()
        import math
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class CpyModel(nn.Module):
    """CPY: F32→F16 dtype cast [2, 1, HEAD_DIM] (KV store-like).

    ExecuTorch QNN backend 가 aten.to.dtype / aten._to_copy 를 독립 op 으로
    delegate 하지 않음 (F16 output tensor QNN_TENSOR_TYPE_MAP KeyError).
    빌드 시도 → 728 B stub (QNN delegation 0 op) → ✗ 마킹 대상.
    paper 에: "CPY (dtype cast) ExecuTorch QNN backend 미지원 — CPU fallback"
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F32 → F16 cast: aten._to_copy.default
        return x.to(torch.float16)


class SetRowsModel(nn.Module):
    """SET_ROWS: KV scatter [2, 1, HEAD_DIM] → cache[2, MAX_CTX, HEAD_DIM].

    torch.index_put_ 로 구현. ExecuTorch export 에서 scatter 로 변환 예정.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        cache: torch.Tensor,   # [2, MAX_CTX, HEAD_DIM]
        src: torch.Tensor,     # [2, 1, HEAD_DIM]
        pos: torch.Tensor,     # [1] long — scatter position
    ) -> torch.Tensor:
        # index_put scatter: cache[:, pos[0], :] = src[:, 0, :]
        idx = pos[0].item() if not torch._dynamo.is_compiling() else pos[0]
        return cache.index_copy(1, pos, src)


# ---------------------------------------------------------------------------
# example_inputs 생성
# ---------------------------------------------------------------------------


def make_inputs(op: str, shape_id: str | None = None) -> tuple:
    torch.manual_seed(42)
    if op == "MUL_MAT":
        k, n = MULMAT_SHAPES[shape_id or "mm_ffn"]
        return (torch.randn(1, k),)
    elif op == "RMS_NORM":
        return (torch.randn(1, HIDDEN),)
    elif op == "ROPE":
        q = torch.randn(1, N_HEADS, HEAD_DIM)
        pos = torch.zeros(1, dtype=torch.long)
        return (q, pos)
    elif op == "FLASH_ATTN_EXT":
        q = torch.randn(1, N_HEADS, 1, HEAD_DIM)
        k = torch.randn(1, N_KV_HEADS, MAX_CTX, HEAD_DIM)
        v = torch.randn(1, N_KV_HEADS, MAX_CTX, HEAD_DIM)
        return (q, k, v)
    elif op == "GET_ROWS":
        return (torch.zeros(1, dtype=torch.long),)
    elif op == "SILU":
        return (torch.randn(1, FFN_DIM),)
    elif op == "MUL":
        return (torch.randn(1, FFN_DIM), torch.randn(1, FFN_DIM))
    elif op == "ADD":
        return (torch.randn(1, HIDDEN), torch.randn(1, HIDDEN))
    elif op == "SOFT_MAX":
        return (torch.randn(N_HEADS, 1, MAX_CTX),)
    elif op == "SCALE":
        return (torch.randn(N_HEADS, 1, MAX_CTX),)
    elif op == "CPY":
        return (torch.randn(2, 1, HEAD_DIM),)  # F32 input → F16 output
    elif op == "SET_ROWS":
        cache = torch.zeros(2, MAX_CTX, HEAD_DIM)
        src = torch.randn(2, 1, HEAD_DIM)
        pos = torch.zeros(1, dtype=torch.long)
        return (cache, src, pos)
    else:
        raise ValueError(f"unknown op: {op!r}")


def make_model(op: str, shape_id: str | None = None) -> nn.Module:
    if op == "MUL_MAT":
        k, n = MULMAT_SHAPES[shape_id or "mm_ffn"]
        return MatMulModel(k, n)
    elif op == "RMS_NORM":
        return RMSNormModel()
    elif op == "ROPE":
        return RoPEModel()
    elif op == "FLASH_ATTN_EXT":
        return FlashAttnModel()
    elif op == "GET_ROWS":
        # vocab=1024 대표 (full vocab 151936 은 OOM 위험)
        return GetRowsModel(vocab=1024)
    elif op == "SILU":
        return SiluModel()
    elif op == "MUL":
        return MulModel()
    elif op == "ADD":
        return AddModel()
    elif op == "SOFT_MAX":
        return SoftMaxModel()
    elif op == "SCALE":
        return ScaleModel()
    elif op == "CPY":
        return CpyModel()
    elif op == "SET_ROWS":
        return SetRowsModel()
    else:
        raise ValueError(f"unknown op: {op!r}")


# ---------------------------------------------------------------------------
# PTE 빌드 코어
# ---------------------------------------------------------------------------


def build_pte(
    op: str,
    shape_id: str | None,
    dtype: str,
    out_path: str,
    verbose: bool = True,
) -> dict:
    """단일 cell PTE 빌드. 반환: {ok, path, size, sha256, elapsed, note}."""
    cell_id = f"{op}{'_' + shape_id if shape_id else ''}_{dtype}"
    t0 = time.time()
    result: dict = {"cell_id": cell_id, "ok": False, "path": out_path, "note": ""}

    # CPY: ExecuTorch QNN backend 미지원 (aten._to_copy / aten.to.dtype 단독 → QNN_TENSOR_TYPE_MAP KeyError)
    # → build_pte 호출 skip, ✗ 마킹 반환
    if op == "CPY":
        result["note"] = "CPY (dtype cast) ExecuTorch QNN backend 미지원: aten._to_copy standalone not delegated (QNN_TENSOR_TYPE_MAP F16 KeyError). ✗ 마킹 — CPU fallback only."
        result["elapsed"] = 0.0
        result["size"] = 0
        result["sha256"] = ""
        if verbose:
            print(f"  [✗]  {cell_id:50s}  SKIP (QNN unsupported)  CPY dtype cast")
        return result

    try:
        torch.manual_seed(42)
        model = make_model(op, shape_id).eval()
        inputs = make_inputs(op, shape_id)

        if dtype == "use_fp16":
            backend_options = generate_htp_compiler_spec(use_fp16=True)
            compiler_spec = generate_qnn_executorch_compiler_spec(
                soc_model=SOC,
                backend_options=backend_options,
            )
            edge = to_edge_transform_and_lower_to_qnn(
                module=model,
                inputs=inputs,
                compiler_specs=compiler_spec,
            )

        elif dtype == "use_8a4w":
            backend_options = generate_htp_compiler_spec(use_fp16=False)
            compiler_spec = generate_qnn_executorch_compiler_spec(
                soc_model=SOC,
                backend_options=backend_options,
            )
            quantizer = QnnQuantizer()
            quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a4w)
            exported = torch.export.export(model, inputs, strict=True).module()
            prepared = prepare_pt2e(exported, quantizer)
            # calibration: single random sample (seed=42 이미 설정)
            prepared(*inputs)
            converted = convert_pt2e(prepared)
            edge = to_edge_transform_and_lower_to_qnn(
                module=converted,
                inputs=inputs,
                compiler_specs=compiler_spec,
            )
        else:
            raise ValueError(f"unknown dtype: {dtype!r}")

        et = edge.to_executorch()
        save_pte_program(et, out_path)

        elapsed = time.time() - t0
        fsize = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        sha = _sha256(out_path)
        result.update(ok=True, size=fsize, sha256=sha, elapsed=elapsed)
        if verbose:
            print(f"  [OK] {cell_id:50s}  {fsize/1024:8.1f} KB  {elapsed:.1f}s  {sha[:12]}...")

    except Exception as exc:
        elapsed = time.time() - t0
        note = str(exc)[:200]
        result.update(ok=False, elapsed=elapsed, note=note, size=0, sha256="")
        if verbose:
            print(f"  [✗]  {cell_id:50s}  FAIL  {elapsed:.1f}s  {note[:80]}")

    return result


def _sha256(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# cell 목록 생성
# ---------------------------------------------------------------------------


def enumerate_cells(
    ops: list[str] | None = None,
    shapes: list[str] | None = None,
    dtypes: list[str] | None = None,
) -> list[tuple[str, str | None, str]]:
    """(op, shape_id, dtype) 튜플 목록. MUL_MAT 은 3 shape 확장."""
    target_ops = ops or ALL_OPS
    target_dtypes = dtypes or ALL_DTYPES
    target_shapes = shapes or list(MULMAT_SHAPES.keys())

    cells: list[tuple[str, str | None, str]] = []
    for op in target_ops:
        for dtype in target_dtypes:
            if op == "MUL_MAT":
                for shape_id in target_shapes:
                    cells.append((op, shape_id, dtype))
            else:
                cells.append((op, None, dtype))
    return cells


def cell_to_path(op: str, shape_id: str | None, dtype: str, out_dir: str) -> str:
    name = op.lower()
    if shape_id:
        name += f"_{shape_id}"
    # use_fp16 → fp16, use_8a4w → w4a8
    suffix = "fp16" if dtype == "use_fp16" else "w4a8"
    return os.path.join(out_dir, f"{name}_{suffix}.pte")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Qwen 2.5-1.5B PTE sweep builder (P1c)",
    )
    p.add_argument(
        "--op",
        choices=ALL_OPS + ["ALL"],
        default="ALL",
        help="단일 op 지정 (default: ALL)",
    )
    p.add_argument(
        "--shape",
        choices=list(MULMAT_SHAPES.keys()) + ["ALL"],
        default="ALL",
        help="MUL_MAT shape (mm_ffn/mm_lmh/mm_qkv, default: ALL)",
    )
    p.add_argument(
        "--dtype",
        choices=ALL_DTYPES + ["ALL"],
        default="ALL",
        help="dtype (use_fp16/use_8a4w, default: ALL)",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "pte"),
        help="출력 디렉토리 (default: ./pte/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="venv 가용성 + 1-cell dry-run (MUL_MAT mm_ffn use_fp16)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=True,
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dry_run:
        print("=== DRY-RUN: MUL_MAT mm_ffn use_fp16 ===")
        out = os.path.join(args.out_dir, "dryrun_mulmat_mm_ffn_fp16.pte")
        r = build_pte("MUL_MAT", "mm_ffn", "use_fp16", out)
        print(f"\nDRY-RUN result: {'OK' if r['ok'] else 'FAIL'}")
        if not r["ok"]:
            print(f"  error: {r['note']}")
            return 1
        print(f"  size: {r.get('size', 0)/1024:.1f} KB")
        print(f"  sha256: {r.get('sha256', '')[:16]}...")
        print(f"  elapsed: {r.get('elapsed', 0):.1f}s")
        return 0

    # cell 필터링
    ops_filter = None if args.op == "ALL" else [args.op]
    shapes_filter = None if args.shape == "ALL" else [args.shape]
    dtypes_filter = None if args.dtype == "ALL" else [args.dtype]

    cells = enumerate_cells(ops_filter, shapes_filter, dtypes_filter)
    total = len(cells)
    print(f"=== PTE SWEEP: {total} cells → {args.out_dir} ===\n")

    results = []
    t_all = time.time()
    for i, (op, shape_id, dtype) in enumerate(cells, 1):
        out_path = cell_to_path(op, shape_id, dtype, args.out_dir)
        label = f"[{i}/{total}]"
        print(f"{label} building {op}{'/' + shape_id if shape_id else ''} {dtype} ...")
        r = build_pte(op, shape_id, dtype, out_path, verbose=args.verbose)
        results.append(r)

    # 요약
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    elapsed_total = time.time() - t_all

    print(f"\n=== SUMMARY ===")
    print(f"  Total : {total}")
    print(f"  OK    : {len(ok)}")
    print(f"  FAIL  : {len(fail)}")
    print(f"  Time  : {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")
    print()
    if ok:
        print("  Built:")
        for r in ok:
            print(f"    {r['cell_id']:50s}  {r.get('size',0)/1024:8.1f} KB  {r.get('sha256','')[:12]}...")
    if fail:
        print("\n  Failed (✗ 마킹 대상):")
        for r in fail:
            print(f"    {r['cell_id']:50s}  {r['note'][:100]}")

    # 결과 JSON 저장
    import json
    summary_path = os.path.join(args.out_dir, "..", "pte_build_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total": total,
                "ok_count": len(ok),
                "fail_count": len(fail),
                "elapsed_sec": round(elapsed_total, 1),
                "cells": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Summary JSON: {summary_path}")
    return 0 if len(fail) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
