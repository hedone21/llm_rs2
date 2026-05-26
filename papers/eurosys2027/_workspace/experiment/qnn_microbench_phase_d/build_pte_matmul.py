#!/usr/bin/env python3
"""build_pte_matmul.py — Qwen FFN gate MatMul .pte 빌드 (Executorch QNN HTP).

본 매트릭스 의 M6b (F16) + M7 (W8A8) cell 용 .pte 파일 생성.

전제:
- Executorch venv 활성화 (`/home/go/Workspace/executorch/.venv`)
- QNN_SDK_ROOT 환경변수 설정 (예: `/home/go/Workspace/llm_rs2/third_party/qnn_sdk_2.33`)
- SoC: SM8750 (Snapdragon 8 Elite, HtpArch.V79)

HTP 는 FP32 native 가속 안 함. M6 (FP32) cell 은 제외.

Output:
- f16  → matmul_f16.pte
- w8a8 → matmul_w8a8.pte
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

K = 1536
N = 8960


class MatMulModel(nn.Module):
    """Qwen 2.5-1.5b FFN gate matmul: y[N] = x[K] @ W[K, N]."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(K, N, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build(variant: str, out_path: str) -> None:
    torch.manual_seed(42)
    model = MatMulModel().eval()
    example_inputs = (torch.randn(1, K),)
    soc = QcomChipset.SM8750

    if variant == "f16":
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=soc,
            backend_options=backend_options,
        )
        edge = to_edge_transform_and_lower_to_qnn(
            module=model,
            inputs=example_inputs,
            compiler_specs=compiler_spec,
        )
    elif variant == "w8a8":
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=soc,
            backend_options=backend_options,
        )
        quantizer = QnnQuantizer()  # default: kHtpBackend + SM8750
        quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a8w)
        m = torch.export.export(model, example_inputs, strict=True).module()
        m = prepare_pt2e(m, quantizer)
        # 단일 random calibration sample. paper-grade 가 아니면 충분.
        m(*example_inputs)
        m = convert_pt2e(m)
        edge = to_edge_transform_and_lower_to_qnn(
            module=m,
            inputs=example_inputs,
            compiler_specs=compiler_spec,
        )
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    et = edge.to_executorch()
    save_pte_program(et, out_path)
    print(f"saved: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["f16", "w8a8"])
    p.add_argument("--out", default=None, help="output .pte path")
    args = p.parse_args()
    out = args.out or f"matmul_{args.variant}.pte"
    build(args.variant, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
