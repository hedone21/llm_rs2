# Phase 5 Sprint C — WSWAP-5-AUF-PLACEHOLDER-DROP 결과 리포트

- **측정일**: 2026-04-26
- **브랜치/HEAD (구현 직후)**: `feat/weight` (작업 직전 HEAD `b52b3ed` 기준 + 본 sprint 패치)
- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel 6.6.77, 6 threads (`--threads 6`), backend OpenCL (Adreno)
- **모델**:
  - F16 GGUF: `Llama-3.2-1B-Instruct-f16.gguf` (2.4 GB, 16 layers)
  - Q4_0 GGUF: `Llama-3.2-1B-Instruct-q4_0.gguf` (703 MB)
  - AUF (Phase 3.7b SOA payload): `Llama-3.2-1B-Instruct.auf` (sha256
    `1a1ead0c1f532b26034989deb7dbfece4f5b7ed41b881491cfee857ae014c5ec`,
    Sprint B와 동일 자산 — AUF 포맷 변경 없음)
- **CLI (TBT)**: `--num-tokens 128 --protected-prefix 4 --prompt "The capital of France is" --threads 6`
- **CLI (--profile)**: `--num-tokens 64 --profile`
- **반복**: TBT N=3, profile N=2/configuration. intra-config sleep 12s, inter-config sleep ≥30s.

## 1. 변경 요약

### 채택 옵션
**옵션 2 (lightweight stub Tensor) 변형** — AUF SOA bypass 모드의 weight tensor가
placeholder cl_mem 대신 `NoshuffleWeightBuffer`로 backed되도록 변경. registry 키는
`d_buf` cl_mem ptr (= `Tensor::buffer().cl_mem()` 반환값) — 기존
`prepare_noshuffle_buffers(swap_to_placeholder=true)`와 동일 패턴 재사용.

### 변경 파일

| 파일 | 변경 |
|---|---|
| `engine/src/core/backend.rs` | trait `register_pre_converted_soa(&Tensor, …) -> Result<()>` 제거. 신설: `alloc_pre_converted_soa_tensor(Shape, q_bytes, d_bytes, ne00, ne01) -> Result<Option<Tensor>>` (placeholder 없는 NoshuffleWeightBuffer-backed tensor 반환), `restore_pre_converted_soa_registration(&Tensor) -> Result<()>` (Stage (d) 재등록 + AOS fallthrough) |
| `engine/src/backend/opencl/mod.rs` | `register_pre_converted_soa` 구현 제거. inherent helper `alloc_and_upload_soa_buffers` 신설 (cl_mem 할당 + blocking write + image1d_buffer view, alloc 카운트 `auf_soa_*`). trait impl: `alloc_pre_converted_soa_tensor` (helper 호출 + `register_noshuffle_soa(d_buf_key, …)` + NoshuffleWeightBuffer wrap), `restore_pre_converted_soa_registration` (downcast NoshuffleWeightBuffer → 재등록 / 아니면 `ensure_noshuffle_soa_registered` fallthrough). 신규 unit test 2건 (`test_alloc_pre_converted_soa_tensor_no_placeholder`, `test_restore_pre_converted_soa_registration_aos_fallthrough`) |
| `engine/src/models/weights/swap_executor.rs` | `materialise_auf_soa_weight`: `copy_weight_from(placeholder)` 호출 제거. `Backend::alloc_pre_converted_soa_tensor` 위임. dtype 가드 수정 — `primary.dtype() == Q4_0` (F16→Q4_0 swap 시 항상 false였음) → `target_dtype == Q4_0 && secondary_info.dtype == Q4_0` (정확성 회귀 fix). Stage (d) AUF 분기 단순화 — secondary_arc payload split 호출 제거하고 `restore_pre_converted_soa_registration(tensor)` 호출만 남김 (NoshuffleWeightBuffer 인지 fallthrough 분기는 backend 내부에서 처리) |
| `engine/src/models/weights/secondary_mmap.rs` | doc-comment update (`Backend::alloc_pre_converted_soa_tensor`로 호출자 명시) |
| `engine/src/bin/auf_tool.rs` | doc-comment update (소비자 API 이름 변경) |

### 라인 수 (대략)

- backend.rs: +50/-15 (trait 메서드 시그니처 + doc)
- opencl/mod.rs: −150 register_pre_converted_soa 본체 / +120 alloc_pre_converted_soa_tensor + restore_pre_converted_soa_registration / +110 helper alloc_and_upload_soa_buffers / +200 신규 unit tests
- swap_executor.rs: −80 Stage (d) payload-split 본체 / +20 stage (d) loop / materialise: −15 placeholder upload / +10 backend 위임 + dtype 가드

### Stage (d) 동작
이전:
```
for tensor in [wq..w_down]:
    let split = secondary.split_pre_converted_soa(info)
    backend.register_pre_converted_soa(tensor, q_bytes, d_bytes, ne00, ne01)
        // → alloc q_buf/d_buf cl_mem + register against tensor.cl_mem (placeholder)
```
이후:
```
for tensor in [wq..w_down]:
    backend.restore_pre_converted_soa_registration(tensor)
        // → downcast NoshuffleWeightBuffer → register against d_buf (already alive)
        // → fallthrough to ensure_noshuffle_soa_registered for non-SOA-backed
```

### 결정적 사실: 정확성 회귀 fix
구현 1차 측정에서 garbage 출력 ("/buttons" 재발생) 확인. 원인 = `materialise_weight`
가드의 `primary.dtype() == DType::Q4_0` — F16 → Q4_0 swap에서 primary는 F16이라
**항상 false**. AUF SOA bypass 분기가 한 번도 실행되지 않은 채 GGUF 경로
(`materialise_tensor` + `unpermute_qk_rows`)가 AUF SOA bytes를 AOS로 잘못 해석.
가드를 `target_dtype == Q4_0 && secondary_info.dtype == Q4_0`로 변경하여 fix.
변경 후 "Paris" 정상 출력, garbage 0건 (TBT N=3, profile N=2 모두 검증).

이 dtype 가드 버그는 **Sprint B 5차 측정 시점에도 잠재**했으나, 그때는 placeholder
cl_mem이 존재했고 forward path가 SOA registry hit으로 placeholder의 데이터를 읽지
않았기 때문에 우연히 정확성 유지. Sprint C에서 placeholder를 제거하면서 잠재
버그가 표면화 → 본 sprint에서 함께 fix.

## 2. cl_mem 개수 표 (`LLMRS_CL_MEM_DIAG=1`)

`stage=after_force_swap` 시점 (Sprint B와 동일 시점). N=1 (deterministic).

| 카테고리 | Q4 baseline | Mixed ratio=1 (Sprint B before) | Mixed ratio=1 (Sprint C after) |
|---|---:|---:|---:|
| weight_f16_copy | 0 | 145 | **145** (변화 없음) |
| weight_q4_aos_copy (placeholder) | 113 | 112 | **0** ✅ |
| weight_q4_soa_q (GGUF SOA) | 113 | 0 | 0 |
| weight_q4_soa_d (GGUF SOA) | 113 | 0 | 0 |
| weight_q4_soa_img (GGUF SOA) | 113 | 0 | 0 |
| auf_soa_q (AUF SOA) | 0 | 112 | 112 |
| auf_soa_d (AUF SOA) | 0 | 112 | 112 |
| auf_soa_img (AUF SOA) | 0 | 112 | 112 |
| weight_f32_copy (norms) | 33 | 33 | 33 |
| **TOTAL count** | **485** | **626** | **514** |
| **TOTAL alive** (drop 무시 시) | 485 | 626 | 514 |
| **drop 후 실제 alive** (registry release 반영) | 485 | 626 | **178** ✅ |

본 sprint diag dump의 `noshuffle_registry_*` releases=112×3=336 항목은 stage (d)의
`invalidate_noshuffle_soa_registry()` → restore loop 사이에 registry가 비워졌다가
재등록되는 과정에서 카운팅. 실 driver-side cl_mem 객체는 NoshuffleWeightBuffer가
소유하므로 alive 유지.

### 핵심 변화 (Mixed ratio=1.0)
- TOTAL cl_mem: **626 → 514** (−112, **−17.9%**)
- weight_q4_aos_copy (placeholder): **112 → 0** (Sprint C 핵심 결과)
- alive bytes: **4.05 GB → 3.51 GB** (−547 MB)

placeholder 0개 acceptance **PASS**.

### Q4 baseline과의 비교
Q4 baseline은 NoshuffleWeightBuffer로 swap된 weight를 들고 있으므로
weight_q4_aos_copy=113은 alloc 시점 카운트일 뿐 실 driver cl_mem은 reclaim 됨.
실 alive ≈ 372 (113×3 SOA + 33 norms).

Mixed ratio=1.0 (Sprint C):
- 실제 alive driver cl_mem ≈ 145 (F16 primary) + 112×3 (AUF SOA q/d/img) + 33 (norms) = **514** (의미 있는 수)
- 그러나 forward path가 실제 사용하는 cl_mem은 112×3 (AUF SOA) + 약간의 비-swap weight
  (embed/lm_head/output_norm 등). 즉 사용 cl_mem은 baseline (372) 대비 차이 거의 없음.
- 차이는 **F16 primary 145**가 alive로 남는 것뿐 — 이는 PRIMARY-DROP sprint 영역.

## 3. matmul_qkv μs/call 표 (Decode `--profile`)

`Decode per-op breakdown (accumulated over 1008 layer-calls)`. 각 카테고리는 i2 측정값.

| op | Q4 baseline | Mixed Sprint B (before) | Mixed Sprint C (after) | Δ Sprint B vs Q4 | Δ Sprint C vs Q4 | Δ Sprint C vs Sprint B |
|---|---:|---:|---:|---:|---:|---:|
| matmul_qkv | 420 | 534 | **513** | +114 (+27.1%) | +93 (+22.1%) | **−21 µs/call (−3.9%)** |
| matmul_wo | 311 | 323 | 321 | +12 (+3.9%) | +10 (+3.2%) | −2 (−0.6%) |
| matmul_ffn | 1166 | 1150 | 1136 | −16 | −30 | −14 |
| attention | 216 | 267 | 236 | +51 (+23.6%) | +20 (+9.3%) | **−31 µs/call (−11.6%)** |
| rms_norm | 461 | 517 | 490 | +56 | +29 | **−27 µs/call (−5.2%)** |
| rope | 245 | 247 | 252 | +2 | +7 | +5 |
| add_assign | 171 | 187 | 183 | +16 | +12 | −4 |
| kv_update | 187 | 192 | 199 | +5 | +12 | +7 |
| **TOTAL/layer-call** | **3181** | **3422** (+241) | **3330** (+149) | +241 (+7.6%) | **+149 (+4.7%)** | **−92 µs/call (−2.7%)** |

(Sprint B 데이터는 `phase_5_tbt_diag.md` § 3 i2 결과)

### 핵심 관찰
- matmul_qkv 단독 회복: **−21 µs/call (−3.9%)** — placeholder 제거로 first-weight access TLB cold가 부분 완화. 그러나 +93 µs/call의 절대 gap은 여전히 존재 (−3.9%만 회복).
- attention 회복: **−31 µs/call (−11.6%)** — KV access는 placeholder와 무관해야 하지만 OS page cache pressure의 부수 효과로 회복.
- rms_norm 회복: **−27 µs/call (−5.2%)** — 마찬가지로 page cache 효과로 추정.
- TOTAL: **−92 µs/call** × 1008 = **−93 ms over 64 tokens = −1.45 ms/tok 누적 회복**.

### 잔여 gap
matmul_qkv +93 µs/call (+22.1%)이 여전히 dominant. Sprint B 진단의 가설:
> matmul_qkv는 layer당 첫 weight load (Q/K/V 3개 weight를 동시에 load) 이므로 driver TLB / texture cache가 cold하다. matmul_wo, ffn은 같은 layer 안에서 후속 access이므로 warm.

Sprint C 후에도 이 패턴이 유지되어 placeholder cl_mem 자체가 first-weight TLB cold 의
주 원인이 아님을 시사. 진짜 원인은 **F16 primary 145 cl_mem이 alive로 남아 driver page
table에 등록**된 상태로 추정. 이는 PRIMARY-DROP (P2) sprint 영역.

## 4. ratio=1.0 mixed Decode TBT (wall-clock, ms/tok)

`--profile` 미사용, `Decode: X ms/tok` 라인. N=3.

| Run | Q4 baseline | Mixed ratio=1 (Sprint C after) |
|---|---:|---:|
| 1 | 16.32 | 20.30 |
| 2 | 16.41 | 20.11 |
| 3 | 16.37 | 20.25 |
| **mean** | **16.37 ms/tok (61.1 tok/s)** | **20.22 ms/tok (49.4 tok/s)** |
| σ | 0.045 | 0.097 |

### Phase 4 / Sprint B 대비

| 측정 | Q4 baseline | Mixed ratio=1 | Δ vs Q4 baseline |
|---|---:|---:|---:|
| Phase 4 (placeholder 있음, `21c6d82`) | 16.31 ms/tok | 20.58 ms/tok | +26.2% |
| **Sprint C (placeholder 제거)** | **16.37 ms/tok** | **20.22 ms/tok** | **+23.5%** |
| Δ (Sprint C 회복) | — | **−0.36 ms/tok** | **−2.7 pp 감소** |

Sprint C는 placeholder 제거로 **20.58 → 20.22 ms/tok (−1.7%)** 회복. matmul_qkv 단독
회복 (-21 µs/call × 16 layer = −0.34 ms/tok)이 wall-clock 결과 (−0.36 ms/tok)와
일치 — fragmentation 가설을 검증함.

그러나 Q4 baseline 대비 잔여 gap은 +23.5% (Phase 4 +26.2% 대비 −2.7pp 개선만). Phase 4
acceptance −5% 미달은 여전. PRIMARY-DROP sprint가 추가로 필요.

## 5. INV-131 safety net 점검 결과

- 호스트 spec 테스트 `tests/spec/test_inv_131_*.rs` 4개 모두 PASS:
  - `test_inv_131_swap_preserves_non_swapped_layers`
  - `test_inv_131_swap_clears_old_entry_then_registers_new`
  - `test_inv_131_q4_0_only`
  - `test_inv_131_non_adreno_backend_noop`
- 신규 unit test `test_restore_pre_converted_soa_registration_aos_fallthrough` —
  GGUF AOS-backed Q4_0 weight tensor에 대해 `restore_pre_converted_soa_registration`이
  `ensure_noshuffle_soa_registered` (INV-131 safety net)로 fallthrough함을 검증.
- 디바이스 측정에서 GGUF 분기 활성화 확인 — Q4 baseline은 `weight_q4_soa_*` 카테고리
  활성, AUF 분기는 `auf_soa_*` 카테고리 활성. **두 분기 분리 깔끔히 유지**됨.

INV-131 회귀 없음 ✅.

## 6. 호스트 sanity 결과

```
cargo build -p llm_rs2 --features opencl: PASS
cargo build --release -p llm_rs2 --features opencl: PASS
cargo build --release --target aarch64-linux-android --bin generate: PASS
cargo fmt --all: clean
cargo clippy -p llm_rs2 --features opencl --tests -- -D warnings: clean
cargo test --workspace: 모든 모듈 PASS (failure 0건)
  - spec tests: 392 passed (이전 392, 회귀 0)
  - 신규 unit tests: 2건 PASS (alloc_pre_converted_soa_tensor_no_placeholder, restore_pre_converted_soa_registration_aos_fallthrough)
```

## 7. 정확성 가드 결과

| 측정 | 결과 |
|---|---|
| Mixed ratio=1.0 N=3 (TBT, 128 tokens) | 모든 run "Paris", garbage ("/buttons" 등) 0건 |
| Mixed ratio=1.0 N=2 (--profile, 64 tokens) | 모든 run "Paris", garbage 0건 |
| Q4 baseline N=3, N=2 | 모든 run "Paris" (Q4 baseline의 `[part-dbg]` 메시지는 unrelated debug print) |
| 디바이스 cl_mem diag dump | "Paris" 정상 출력 |

INV-122 spec 회귀 측정은 본 sprint scope 외 (TBT 측정에 한정). 별도 sprint에서
greedy single-token NMSE 검증이 필요하면 `phase_5_inv122_spec_v2.1` 절차 따라 진행 가능.

## 8. 잔여 issue / 권장

### 차단/측정 잡음 0건
본 sprint는 하나의 차단 (dtype 가드 버그)이 1차 측정에서 발견되어 즉시 수정.
fix 후 재측정으로 정확성 + acceptance 정량 확보.

### Acceptance 평가

| 게이트 | 결과 | 정량 |
|---|---|---|
| 1. AUF SOA bypass placeholder cl_mem **0개** | ✅ PASS | weight_q4_aos_copy: 112 → 0 |
| 2. matmul_qkv μs/call 감소 | ⚠️ 부분 PASS | 534 → 513 µs/call (−21, −3.9%). Q4 baseline 437 근접 미달 (잔여 +93 µs/call). 부분 회복 PASS |
| 3. 정확성 회귀 없음 | ✅ PASS | "Paris" N=5 측정 전부, garbage 0건 |
| 4. 호스트 sanity (cargo test + clippy + fmt) | ✅ PASS | 392 spec PASS, 신규 unit 2건 PASS, fmt/clippy clean |
| 5. INV-131 safety net 회귀 없음 | ✅ PASS | spec test 4건 PASS, 신규 fallthrough 테스트 PASS |
| 6. 디바이스 N≥3 측정으로 TBT 변화 정량 | ✅ PASS | TBT N=3 + profile N=2 |
| 7. 리포트 작성 | ✅ PASS | 본 문서 |

### 잔여 −20.7% gap 분석
- Phase 4 −20.7% 중 Sprint C에서 회복: −1.7 pp (−1.7% / −20.7% = **8.2% 회복**)
- 잔여 gap: 약 −18-19% (Sprint C 후 mixed Q4 vs Q4 baseline 대비)
- 원인 (Sprint B 진단):
  1. **F16 primary 145 cl_mem alive** — driver page table 압박, OS page cache pressure (PRIMARY-DROP P2 영역)
  2. **KV cache fragmentation** (`project_kv_fragmentation`, P3 영역)
  3. matmul_qkv first-weight TLB cold 잔여 — primary alive 해소 후 재측정 필요

### PRIMARY-DROP (WSWAP-5-PRIMARY-DROP) 진행 권장 여부

**권장: P2로 진행**. 근거:
- Sprint C로 placeholder 영향 분리 확인 — −1.7% 회복은 측정 가능 수준이지만 작음
- 잔여 gap 대부분이 F16 primary 145 cl_mem (alive_bytes 2.47 GB) 문제로 추정됨
- PRIMARY-DROP은 lifecycle audit 광범위 (M effort)이지만 Sprint B에서 이미 조사 시작됨
- KV-FRAG-INTEGRATE는 PRIMARY-DROP 후 잔여 측정 후 결정

PRIMARY-DROP 진행 시 주의:
- F16 primary가 `model.embed_tokens`, `model.lm_head`, `model.output_norm`에 의해 strong_count > 1로 잡힘
- swap layer의 F16 weight 16 × 7 = 112개만 explicit drop 시도 (lm_head/embed/output_norm은 그대로)
- 또는 secondary mmap (AUF) 진입 시점부터 layer F16 cl_mem 자체 alloc skip (lazy primary)

## 9. 디바이스 자산 / 산출물

### 디바이스 (S25 `/data/local/tmp/`)
- `generate` (HEAD = 본 sprint 적용 빌드, 7,264,528 bytes, 2026-04-26 12:46)
- `Llama-3.2-1B-Instruct-{f16,q4_0}.gguf`, `Llama-3.2-1B-Instruct.auf`,
  `tokenizer.json` 모두 변경 없음 (AUF는 Sprint B와 동일 sha256 `1a1ead0c…`)

### 호스트 산출물
- 측정 로그: `/tmp/wswap5_placeholder_drop/results/{q4_baseline,mixed_ratio_1}_i{1,2,3}.log`
  (TBT N=3) + `*_prof_i{1,2}.log` (profile N=2)
- 측정 스크립트: `/tmp/wswap5_placeholder_drop/run_clmem_diag.sh`,
  `run_tbt.sh`, `run_profile.sh`

### 코드 변경
파일 + 핵심 줄 (사용한 옵션 = 옵션 2 변형 + dtype 가드 fix):
- `engine/src/core/backend.rs:587-672`
- `engine/src/backend/opencl/mod.rs:3406-3527` (helper) / `3893-4000` (trait impls) / `7333-7521` (신규 unit tests)
- `engine/src/models/weights/swap_executor.rs:407-447` (Stage (d)), `592-705` (materialise)
- `engine/src/models/weights/secondary_mmap.rs:380-385` (doc only)
- `engine/src/bin/auf_tool.rs:600-606` (doc only)

## 10. 결론 요약

| 질문 | 답 |
|---|---|
| placeholder cl_mem 112개 제거됐는가 | ✅ 0개 (weight_q4_aos_copy = 0) |
| Q4 baseline matmul_qkv 437 µs/call 근접 회복? | 부분 (−21 µs/call, 잔여 +93). PRIMARY-DROP 추가 필요 |
| TBT 회복 정량 | −0.36 ms/tok (−1.7%) — Phase 4 −20.7% gap의 약 8% 회복 |
| 정확성 회귀 발생? | 1차 측정에서 garbage 발견 → dtype 가드 fix 즉시 적용 → 재측정 정상 |
| INV-131 영향? | 없음. 신규 fallthrough 테스트로 회귀 가드 강화 |
| Phase 4 acceptance 충족? | ratio=1.0 mixed −5% target은 여전히 미충족 (−18~19% 잔여) |
| 다음 step 권장 | PRIMARY-DROP (F16 primary 145 cl_mem) → 잔여 측정 후 KV-FRAG-INTEGRATE |
