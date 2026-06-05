#!/usr/bin/env python3
"""Backend × format × phase 성능 측정 드라이버 (S25, argus_cli).

CPU / GPU(opencl) / NPU(htp) × {q4_0, f16} 를 2-point prefill slope +
steady-state decode 로 측정한다.

- prefill tok/s : (len_L - len_S) / ((TTFT_L - TTFT_S)/1000)
    → 짧은/긴 프롬프트 TTFT 차이로 1회성 init(OpenCL JIT, plan build) 상쇄.
- decode tok/s  : 긴 프롬프트 run 의 'Decode: .. (.. tok/s)' (steady state).

env 함정: --backend htp 는 ADSP_LIBRARY_PATH 필수 (DSP skel 탐색).
"""
import json
import re
import subprocess
from pathlib import Path

SERIAL = "R3CY408S5SB"
WORK = "/data/local/tmp"
ENV = (
    "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 "
    "ADSP_LIBRARY_PATH=/data/local/tmp taskset 3f"
)
TOK = "models/qwen2.5-1.5b-gguf/tokenizer.json"
MODELS = {
    "q4": "models/qwen2.5-1.5b-gguf/qwen2.5-1.5b-q4_0.gguf",
    "f16": "Qwen2.5-1.5B-Instruct-f16.gguf",
}
BACKENDS = {"CPU": "cpu", "GPU": "opencl", "NPU": "htp"}
NUM_TOKENS = 48

PROMPTS = {
    "short": "The history of artificial intelligence began in the mid twentieth century.",
    "long": (
        "The history of artificial intelligence began in the mid twentieth century "
        "when researchers started exploring whether machines could simulate aspects "
        "of human reasoning. Early work focused on symbolic logic and search "
        "algorithms, but progress was limited by the computing power of the era. "
        "Decades later the rise of statistical methods and large datasets shifted "
        "the field toward machine learning, and eventually deep neural networks "
        "transformed what was possible. Today large language models trained on vast "
        "text corpora can generate fluent prose, translate between languages, and "
        "answer questions across many domains, raising concerns about reliability."
    ),
}

RE_TOKLEN = re.compile(r"Token Length:\s*(\d+)")
RE_TTFT = re.compile(r"TTFT:\s*([\d.]+)\s*ms")
RE_DECODE = re.compile(r"Decode:\s*([\d.]+)\s*ms/tok\s*\(([\d.]+)\s*tok/s\)")


def run_cell(backend_flag, model, prompt):
    remote = (
        f"cd {WORK} && {ENV} ./argus_cli --backend {backend_flag} "
        f"--model-path {model} --tokenizer-path {TOK} "
        f"--num-tokens {NUM_TOKENS} --greedy --prompt '{prompt}'"
    )
    p = subprocess.run(
        ["adb", "-s", SERIAL, "shell", remote],
        capture_output=True, text=True, timeout=240,
    )
    out = p.stdout + p.stderr
    tl = RE_TOKLEN.search(out)
    tt = RE_TTFT.search(out)
    dc = RE_DECODE.search(out)
    return {
        "token_len": int(tl.group(1)) if tl else None,
        "ttft_ms": float(tt.group(1)) if tt else None,
        "decode_ms_tok": float(dc.group(1)) if dc else None,
        "decode_tps": float(dc.group(2)) if dc else None,
        "ok": bool(tt and dc),
        "raw_tail": out[-400:],
    }


def main():
    results = {}
    for label, bflag in BACKENDS.items():
        for fmt, model in MODELS.items():
            cell = {}
            for plen, prompt in PROMPTS.items():
                print(f"[run] {label}/{fmt}/{plen} ...", flush=True)
                try:
                    r = run_cell(bflag, model, prompt)
                except subprocess.TimeoutExpired:
                    r = {"ok": False, "raw_tail": "TIMEOUT"}
                cell[plen] = r
                tt = r.get("ttft_ms")
                dt = r.get("decode_tps")
                print(f"      toklen={r.get('token_len')} TTFT={tt} ms "
                      f"decode={dt} tok/s ok={r.get('ok')}", flush=True)
            # derive prefill slope + decode
            s, l = cell["short"], cell["long"]
            prefill_tps = None
            if s.get("ok") and l.get("ok"):
                d_tok = l["token_len"] - s["token_len"]
                d_ms = l["ttft_ms"] - s["ttft_ms"]
                if d_ms > 0 and d_tok > 0:
                    prefill_tps = d_tok / (d_ms / 1000.0)
            cell["derived"] = {
                "prefill_tps": prefill_tps,
                "decode_tps": l.get("decode_tps"),
            }
            results[f"{label}/{fmt}"] = cell
            print(f"  => {label}/{fmt}: prefill={prefill_tps} tok/s "
                  f"decode={l.get('decode_tps')} tok/s\n", flush=True)

    outp = Path(__file__).parent / "results.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"written {outp}")


if __name__ == "__main__":
    main()
