# LISWAP-PPL — Layer Swap NLL gap root cause + fix (2026-05-12)

## TL;DR

이전 handoff (`handoff_liswap_ppl_nll_measurement_2026_05_12.md`) 에서 발견된
"swap process artifact" — F16+swap 시나리오 (C) 가 Q4-native (D) 대비
**+0.065 mean NLL** 으로 잔존한 문제 — 의 **root cause 를 isolate** 하고
**default 동작을 수정** 했다.

- 가설 검증 6 시나리오 진행 (E, F, G, H + weight bit-level dump)
- **root cause**: `--quantize-lm-head` 의 기본값 `auto` 가 PPL 모드 +
  `--secondary-gguf` 조합에서 lm_head 를 runtime 에 F16 → Q4_0 으로 derive.
  Q4-native baseline 은 GGUF 의 lm_head 를 F16 그대로 로드하므로 두 시나리오의
  lm_head dtype 이 다름 → 모든 token logit 에 systematic bias.
- **fix**: PPL 모드 + `--secondary-gguf` + `--quantize-lm-head` 명시 안 했을 때
  자동으로 `none` 으로 override + `[Notice]` 출력.
- **검증**: 시나리오 H (lm_head F16 유지) 의 `total_nll = 2389.4077753537395`
  가 시나리오 D 와 **소수점 끝자리까지 일치** (bit-identical PPL).

## 커밋 (이 작업 전 / 이 작업 후)

- 직전: `69b77760` (이전 handoff)
- 이 작업 변경 파일: `engine/src/bin/generate.rs` (대상 변경 4 곳)

## 시나리오 정리

| 시나리오 | 모델 + swap 설정 | lm_head dtype | mean NLL | PPL | 비고 |
|---|---|---|---:|---:|---|
| A | F16 baseline, no swap | F16 | 2.1438 | 8.5320 | — |
| B | F16 → swap 25 layer | **Q4_0 (derived)** | 2.2822 | 9.7984 | quantized lm_head |
| C | F16 → swap 28 layer | **Q4_0 (derived)** | 2.2961 | 9.9355 | quantized lm_head |
| **D** | **Q4_0 native, no swap** | **F16 (GGUF)** | 2.2310 | 9.3092 | baseline 비교 기준 |
| D' | D 재측정 (noise floor) | F16 | 2.2310 | 9.3092 | **D 와 비트 단위 동일** |
| E | F16 → swap 28 + KV reset + re-prefill | **Q4_0 (derived)** | 2.2997 | 9.9716 | cache reset 효과 없음 |
| F | E + pass2 prefill 1072 (batch only) | Q4_0 (derived) | 2.2981 | 9.9549 | path 가설 기각 |
| G | D 재측정 (= D') | F16 | 2.2310 | 9.3092 | OpenCL deterministic 확인 |
| **H** | **E + `--quantize-lm-head none`** | **F16 (preserved)** | **2.2310** | **9.3092** | **D 와 비트 단위 동일** |

## 가설 검증 추론 체인

1. **시나리오 E (KV cache reset)**: swap 완료 후 KV cache 0 으로 reset + prefill
   다시 시작 → 결과 **PPL 9.97**, C 와 동일. → KV cache mismatch 가설 **기각**.
2. **시나리오 G (D vs D' 재현)**: 두 D run 의 per-token NLL 이 **std = 0** (비트 단위
   동일). → OpenCL non-determinism 가설 **기각**. noise floor = 0 이므로 +0.07 차이는
   100% systematic.
3. **시나리오 F (batch-only prefill 1072)**: pass 2 가 prefill 만 (decode 0) → 결과
   **PPL 9.95**, E 와 동일 + D 와 +0.07. → "prefill vs decode kernel divergence"
   가설 **기각**.
4. **weight bit-level dump (D vs E, 196 file)**: 28 layer 의 7 weight tensor (wq, wk,
   wv, wo, w_gate, w_up, w_down) 합 737 MB 를 byte-level 비교 → **100% byte-equal,
   0 diff byte**. → "swap path 의 quantization 미세 차이" 가설 **기각**.
5. **model-level tensor dump**: dump 함수에 `embed_tokens`, `norm`, `lm_head` 추가
   → **lm_head dtype mismatch 발견**:
   - D: F16 (466747392 bytes)
   - E: **Q4_0 (131272704 bytes)**
6. **시나리오 H (E + lm_head F16 유지)**: `--quantize-lm-head none` → `total_nll`
   소수점 끝자리까지 D 와 일치. → **lm_head Q4_0 derive 가 root cause 확정**.

## Root cause 상세

### Default `auto` 동작 (수정 전)

`engine/src/bin/generate.rs` 의 `--quantize-lm-head` 옵션:
- default: `"auto"`
- 동작 (PPL 모드 + `--secondary-gguf` 조합):
  ```
  if secondary_gguf.is_some():
    if secondary is AUF and has lm_head Q4_0 entry → load directly
    if secondary is AUF without entry → runtime quantize (F16 → Q4_0)
    if secondary is plain GGUF                → runtime quantize (F16 → Q4_0)
  if secondary_gguf.is_none(): → no action (F16 preserved)
  ```

### 시나리오별 lm_head dtype

| 시나리오 | secondary_gguf | qlm 분기 | lm_head 최종 dtype |
|---|---|---|---|
| D (Q4 native) | 없음 | match `_` → no action | **F16** (GGUF 파일 그대로) |
| E (F16 + swap) | qwen2.5-1.5b-q4_0.gguf | `NotAuf` → runtime quantize | **Q4_0** (derive) |

두 시나리오 모두 28 transformer layer 의 weight 는 GGUF Q4_0 모델에서 와서 byte-equal.
그러나 lm_head 만 다른 정밀도 → 모든 token logit 에 +~0.07 NLL bias.

### Why GGUF Q4_0 모델의 lm_head 는 F16?

`qwen2.5-1.5b-q4_0.gguf` 의 quantization 정책: transformer layer 의 큰 weight
(wq/wk/wv/wo/MLP) 는 Q4_0 로 양자화하고, embed_tokens / lm_head / norm 등 작은
보조 weight 는 F16 으로 유지하는 것이 GGUF convention. 따라서 D 의 lm_head 는
원본 F16 그대로.

### Why E 의 derived Q4_0 ≠ D 의 F16?

- D 의 lm_head F16 logit 은 full precision matmul → 정밀한 softmax → 정확한 NLL.
- E 의 derived Q4_0 lm_head 는 block-wise quantized (block=32) → 작은 dequant
  rounding error 가 모든 vocab 행에 누적 → systematic logit shift → +0.07 NLL.

## 코드 변경 (engine/src/bin/generate.rs)

### (1) PPL warmup-swap + measure-prefill 옵션

`--ppl-warmup-swap` (시나리오 E 인프라) + `--ppl-measure-prefill-tokens` (pass 2
prefill 길이 별도 지정, 시나리오 F 인프라). caller 에서 pass 1 (warmup_only=true)
→ KV cache reset → pass 2 (measurement) 의 2-단계 측정 path 구현.
`run_ppl` 시그니처에 `warmup_only: bool` 추가, `plan_done` 시점에 early return.

### (2) Q4 weight readback dump

`--dump-q4-after-load <dir>` / `--dump-q4-after-swap <dir>` 두 옵션 추가.
`fn dump_layer_weights_to_dir` 가 28 layer × 7 weight tensor + model.embed_tokens
/ model.norm / model.lm_head 를 readback 해서 `<dir>/layer{NN}_{name}_{dtype}.bin`
및 `<dir>/model_{name}_{dtype}.bin` 으로 dump. OpenCL backend 에서는 cl_mem 핸들
경유 `read_buffer` 가 device → host copy. CPU-resident tensor (lm_head_on_cpu)
는 CpuBackend fallback.

### (3) Default lm_head quantize 동작 수정 (root fix)

```rust
let qlm = {
    let raw = args.quantize_lm_head.to_ascii_lowercase();
    if args.ppl.is_some()
        && args.secondary_gguf.is_some()
        && (raw == "auto" || raw.is_empty())
    {
        eprintln!(
            "[Notice] PPL mode + --secondary-gguf: auto-disabling lm_head Q4_0 \
             quantization (would create a systematic +~0.07 NLL gap vs Q4-native \
             baseline). Pass `--quantize-lm-head q4_0` to override."
        );
        "none".to_string()
    } else {
        raw
    }
};
match qlm.as_str() { ... }
```

PPL 모드 + secondary GGUF + 명시 override 안 함 → 자동 `none` + 경고. 사용자가
의도적으로 lm_head Q4_0 효과를 측정하려면 `--quantize-lm-head q4_0` 명시.

### (4) Args derive Clone (caller 가 args 두 번 호출하기 위해)

`#[derive(Parser, Debug, Clone)]` 으로 `Clone` 추가. `--ppl-warmup-swap` caller 가
warmup pass 와 measure pass 에 다른 인자를 넘기기 위함 (`ppl_swap_at_token = None`
on pass 2 등).

## 재현 명령 (Galaxy S25 R3CY408S4HN)

### 검증: F16 + swap 28 layer 가 D 와 비트 단위 동일 PPL

```bash
adb -s R3CY408S4HN shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
  LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1 \
  ./generate -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --secondary-gguf models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --backend opencl --temperature 0 \
    --ppl ppl_ref.txt --ppl-prefill-tokens 32 \
    --ppl-swap-at-token 0 --ppl-swap-ratio 1.0 --ppl-swap-per-tick 1 \
    --ppl-warmup-swap"
```

기대: `total_nll = 2389.4077753537395`, `PPL = 9.3092` (D 와 동일).
`[Notice]` 메시지 한 줄 출력.

### Override: 옛 동작 강제 (lm_head Q4_0)

```bash
# Add this flag to the above command:
--quantize-lm-head q4_0
```

기대: `total_nll ≈ 2463`, `PPL ≈ 9.97` (이전 시나리오 E 결과).

## Raw 데이터 위치

### 호스트 (분석용)

```
/tmp/ppl_csv/
├── ppl_A_baseline.csv         시나리오 A
├── ppl_B_standard.csv         시나리오 B (lm_head Q4_0)
├── ppl_C_boundary.csv         시나리오 C (lm_head Q4_0)
├── ppl_D_q4_native.csv        시나리오 D
├── ppl_D2_q4_native.csv       시나리오 D' (재측정, D 와 비트 단위 동일)
├── ppl_E_warmup_reset.csv     시나리오 E (lm_head Q4_0, KV reset)
├── ppl_F_prefill_only.csv     시나리오 F (pass 2 prefill 1072)
└── ppl_H_lmhead_f16.csv       시나리오 H (root fix 검증, D 와 동일)

/tmp/q4_diff/
├── dump_D/      D 모델 28 layer × 7 weight + 3 model tensor (737 MB)
└── dump_E/      E 모드 swap 후 동일 구조 (737 MB)
                 → 196 weight file 모두 byte-equal,
                   model_lm_head dtype 만 다름 (F16 vs Q4_0)
```

### 디바이스

```
R3CY408S4HN:/data/local/tmp/
├── generate                              새 빌드 (이 작업 commit)
├── ppl_E_default.csv                     새 default 동작 검증 출력
└── (D, E, F, H 의 raw CSV 들)
```

## 후속 작업 후보

### 즉시
- **다른 모델 재현성 검증**: Llama 3.2 1B 동일 흐름. lm_head dtype 차이가 다른
  모델에서도 같은 NLL bias 를 만드는지.
- **Partial swap + F16 lm_head 의 quality-efficiency curve**: ratio = {0.25, 0.5,
  0.75, 1.0} 에서 lm_head F16 유지 시 PPL. D 보다 *낮은* PPL 을 노릴 수 있는
  sweet spot 탐색.

### 논문 측면
- 본 finding 은 "Layer Swap 의 quality" 평가 시 lm_head 정밀도 control 이 필수임을
  보임. swap 단독 효과 측정을 위해 lm_head dtype 을 baseline 과 일치시키는 것이
  apples-to-apples.
- 시나리오 E + cache reset 가설 검증 + weight dump 가설 검증 + lm_head dtype
  finding 은 ablation 도구로 강력. 본문에 "ablation: lm_head dtype matters" 1
  단락 + 표 (D vs C vs H) 추가 가능.

### 코드 위생
- AUF v0.1.1 (`lm_head_mode = "auto"` 로 생성된 AUF) 의 lm_head Q4_0 entry 경로 (
  `LmHeadAufResolution::Found`) 도 PPL 모드 + warmup-swap 시 같은 root issue 를
  유발하는지 확인. 만약 그렇다면 위 default override 가 `Found` 경로도 cover
  하도록 확장.
- `--quantize-lm-head` 옵션 자체의 default 를 `none` 으로 바꾸는 것도 고려 (
  PPL 외 generation 모드에도 영향). 현재는 PPL + secondary 조합으로 좁혀 둠.

## 관련 문서

- 이전 handoff: `handoff_liswap_ppl_nll_measurement_2026_05_12.md` (4 시나리오
  + 후속 가설 목록 — 이 문서가 후속 작업 결과)
- LISWAP-6 production: `project_liswap6_alias_production.md` (memory)
- AUF tool: `engine/src/bin/auf_tool.rs` (lm_head Q4_0 entry 변환 path)
- transformer model: `engine/src/models/transformer.rs:472` `quantize_lm_head_to_q4_0`
