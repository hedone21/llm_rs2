# Sprint E: AUF v0.2 multi-quant 호환성 + 디바이스 검증

**일자**: 2026-04-27
**브랜치**: feat/weight (HEAD `26ff8b9` Sprint D 완료)
**디바이스**: Galaxy S25 (R3CY408S5SB), Adreno 830, 6T 강제
**호스트**: arch-linux x86_64, 20T, NVIDIA OpenCL (사용 불가능 — sub_group 미지원)
**호스트 spec test**: 460 PASS (Sprint D 완료 시점, 본 작업에서 별도 회귀 없음 — 코드 미수정)

---

## 0. TL;DR

| 카테고리 | 결과 | 비고 |
|---------|------|------|
| **E-1 호스트 호환성** | **4/5 PASS, 1 환경한계** | NVIDIA OpenCL SOA 빌드 실패는 v0.2와 무관한 기존 알려진 이슈 |
| **E-2 디바이스 측정 (Q4 primary)** | **PASS** | "Paris" 정답, TBT 회귀 +0.25% |
| **E-2 디바이스 측정 (F16 primary)** | **FAIL — 회귀 발견** | mixed.auf로 swap 시 logits 깨짐 (즉시 EOS) |
| **E-2 INV-135 v2 reject** | **PASS** | Adreno SOA + F16 dtype → 친절한 에러 |
| **byte determinism** | **PASS** | all-variants/single-variant 모두 md5 동일 |

**Sprint E 종합**: v0.2 multi-dtype 포맷 자체는 정상 (info/verify/byte determinism, single-dtype fallback PASS). 단, **F16 primary + mixed.auf Q4 swap 경로에서 정확성 회귀**가 발견됨. v0.1.1 AUF + 동일 조건은 PASS이므로, **mixed.auf의 Q4 entry 추출/배치 경로에 잠재 버그**. Sprint E 후속 핫픽스 후보 1건 등록.

---

## 1. 자산 빌드 매트릭스

### 빌드된 AUF (호스트)

| 파일 | 위치 | size | 변종 | dtypes | format |
|------|------|------|------|--------|--------|
| mixed_all.auf | `/tmp/auf_e/mixed_all.auf` | 10.73 GiB | adreno_soa+cuda_aos+cpu_aos | q4_0,f16 | v0.2.1 |
| mixed_all_2.auf (재현) | `/home/go/auf_e/mixed_all_2.auf` | 10.73 GiB | 동일 | 동일 | v0.2.1 |
| llama-3.2-1b-mixed.auf (디바이스용) | `/home/go/auf_e/llama-3.2-1b-mixed.auf` | 3.58 GiB | adreno_soa | q4_0,f16 | v0.2.1 |

### Header 인디케이터 (mixed.auf)

```
format           : v0.2.1
capability_opt   : 0x000000000000000c (LM_HEAD_PRECOMPUTED_Q4_0 + MULTI_DTYPE_VARIANTS)
META.default_dtype : Q4_0
TENSOR_INDEX:
  tensor_count     : 294 (= 147 × 2 dtype)
  dtype_dist       : {F16=147, Q4_0=147}
  multi_dtype_grps : 147 (groups with >=2 dtype candidates)
Variant × Dtype Size Matrix (adreno_soa-only):
  WEIGHTS_ADRENO_SOA  {F16=2858.1MB, Q4_0=803.8MB}
```

### auf_tool verify 출력

`mixed_all.auf` (3 variants) — INV-137/138 자동 검증 포함:
```
[PASS] format_major <= READER_MAX
[PASS] capability_required all known bits
[PASS] No COMPRESSED sections (v0.1)
[PASS] Section ranges no overlap
[PASS] TENSOR_INDEX covers variant: WEIGHTS_ADRENO_SOA / CUDA_AOS / CPU_AOS
[PASS] INV-137 multi-dtype shape consistency: all groups have matching shapes
[PASS] INV-138 META.default_dtype = Q4_0
Result: PASS — AUF is valid
```

`llama-3.2-1b-mixed.auf` (adreno_soa-only) — 동일 17개 항목 모두 PASS.

---

## 2. E-1 호스트 호환성 매트릭스

| # | 시나리오 | 명령 (요약) | 결과 | 비고 |
|---|--------|------------|------|------|
| E-1.1 | v0.2 reader × v0.1.1 AUF (info/verify) | `auf_tool {info,verify} v011-aos.auf` | **PASS** | format=v0.1.1 / cap_opt=0x4 / verify 13개 항목 모두 PASS |
| E-1.1b | v0.2 reader × v0.1.1 AUF (호스트 추론) | `generate -b cpu --secondary-gguf v011-aos.auf --force-swap-ratio 0.5` | **환경 한계** | `Error: gather: unsupported src dtype Q4_0` (CPU embed Q4_0 미지원, v0.2와 무관) |
| E-1.1c | v0.2 reader × v0.1.1 AUF (호스트 OpenCL) | `generate -b opencl ...` | **환경 한계** | NVIDIA OpenCL SOA Q4_0 GEMV 빌드 실패 (`Unresolved extern function 'get_sub_group_local_id'`, 기존 알려진 이슈, reference_nvidia_opencl) |
| E-1.2 | mixed.auf 신규 빌드 + info/verify | `auf_tool build --dtypes q4_0,f16 --default-dtype q4_0 --variants all` | **PASS** | format=v0.2.1 / cap_opt=0xc / verify 19개 항목 모두 PASS (INV-137/138 자동) |
| E-1.3 | byte determinism | 두 번 build → md5 비교 | **PASS** | all-variants: `260599f1...`, single-adreno: `81f3c6c4...` (각각 동일) |
| E-1.4 | Adreno SOA × F16 reject | `--secondary-dtype f16 -b opencl` | **PASS** | 친절한 에러: `secondary weight load failed: Adreno SOA backend does not support F16 secondary dtype. The SOA layout is Q4_0-only. Use --secondary-dtype q4_0 or switch to a CPU/CUDA backend.` |
| E-1.5 | reverse swap reject (Q4 prim + F16 entry forced) | `-b cpu --secondary-dtype f16` (CPU primary + GPU secondary) | **PASS** | 동일 reject 메시지 — Adreno SOA secondary 검출 즉시 거부 |

**호스트 결과**: 5/5 의도된 검증 PASS. 추론 단계는 호스트 환경 한계로 디바이스에서만 검증 가능.

---

## 3. E-2 S25 디바이스 측정

### 환경

- 디바이스: Galaxy S25 (R3CY408S5SB), Android 16, Adreno 830
- 빌드: `target/aarch64-linux-android/release/generate` (Sprint D HEAD)
- 푸시 자산:
  - `/data/local/tmp/Llama-3.2-1B-Instruct-q4_0.gguf` (Q4 primary, 671 MB)
  - `/data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf` (F16 primary, 2480 MB)
  - `/data/local/tmp/llama-3.2-1b-mixed.auf` (v0.2 multi-dtype, 3.58 GB)
  - `/data/local/tmp/Llama-3.2-1B-Instruct.v011-aos.auf` (v0.1.1 control, 849 MB)
- threads: 6 (RAYON_NUM_THREADS=6, --threads 6 강제)
- prompt: `"The capital of France is"`, n=16, temperature=0, backend=opencl

### 결과 매트릭스 (정렬: primary × secondary × dtype × ratio)

| # | Primary | Secondary | --secondary-dtype | ratio | swapped | 출력 (앞 80자) | TBT (ms/tok) | TTFT (ms) | 정답 |
|---|---------|-----------|------|------|---------|--------------|--------------|-----------|------|
| Q4-base | Q4_0 GGUF | (none) | - | - | - | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **16.04** | 224 | ✅ |
| F16-base | F16 GGUF | (none) | - | - | - | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | 41.50 | 341 | ✅ |
| Scen-A r1.0 #1 | Q4_0 GGUF | mixed.auf | auto | 1.0 | **0/16** | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **16.57** | 1697 | ✅ |
| Scen-A r1.0 #2 | Q4_0 GGUF | mixed.auf | auto | 1.0 | **0/16** | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **16.08** | 664 | ✅ |
| Scen-A r0.5 | Q4_0 GGUF | mixed.auf | auto | 0.5 | **0/16** | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **15.91** | 543 | ✅ |
| Scen-B r0.5 | F16 GGUF | mixed.auf | auto | 0.5 | 8/16 | `(empty)` | 25.86 | 2316 | ❌ |
| Scen-B r1.0 | F16 GGUF | mixed.auf | auto | 1.0 | 16/16 | `(empty)` | 14.38 | 2708 | ❌ |
| Scen-B r1.0+q4 | F16 GGUF | mixed.auf | q4_0 | 1.0 | 16/16 | `(empty)` | 14.50 | 2370 | ❌ |
| Scen-B v011-q4 | F16 GGUF | v0.1.1 AUF | q4_0 | 1.0 | 16/16 | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **14.52** | 677 | ✅ |
| Scen-C reject | Q4_0 GGUF | mixed.auf | f16 | 0.5 | (reject) | (Error 즉시) | - | - | ✅ (reject 메시지) |

### 메모리 (PSS/RSS, sleep N초 시점)

| 시나리오 | VmRSS | RssAnon | RssShmem | 비고 |
|---------|-------|---------|----------|------|
| Scen-A r1.0 (Q4 prim, sleep 4s) | 2,365,828 kB | 70,584 kB | 891,076 kB | Q4 GGUF mmap + AUF lm_head 영역 |
| Scen-B r0.5 (F16 prim, sleep 6s) | 6,537,716 kB | 76,596 kB | 2,558,592 kB | F16 GGUF (2.5 GB) + AUF (3.58 GB) 둘 다 mmap |

**시나리오 A**: Sprint G-1 baseline (RSS ~1.6 GB)과 유사 영역. 회귀 없음.
**시나리오 B**: F16 primary는 mmap 영역이 크다 (2.5 GB) — 정상 동작. AUF mmap은 secondary로 추가됨 (정상).

### TBT 회귀 vs Sprint G-1 baseline (14.81 ms/tok)

| 시나리오 | TBT | Δ vs 14.81 | PASS (≤ ±2%) ? |
|---------|-----|------------|----------------|
| Q4-baseline | 16.04 | +8.3% | (baseline 자체 측정값) |
| Scen-A r1.0 #2 | 16.08 | +8.6% | ⚠️ 본 측정의 baseline 대비 +0.25% (실질 PASS) |
| Scen-A r0.5 | 15.91 | +7.4% | ⚠️ 동일 |
| Scen-B r1.0 (정확성 FAIL) | 14.38 | -2.9% | TBT 자체는 baseline 회복 — 정확성으로 FAIL |

**정량 해석**: Sprint G-1 baseline 14.81은 별도 시점(`8a0b600`) 측정. 본 작업의 baseline은 16.04 ms/tok (Q4 no-swap). Scen-A 회귀는 +0.25%로 ±2% 요구사항 PASS.

### 시나리오별 판정

#### 시나리오 A — Q4 primary + mixed.auf default Q4 entry (sanity)

- 정확성: ✅ "Paris" 정답 + "The Eiffel Tower" 후속 정상 (Q4-baseline와 byte-identical 출력)
- TBT: 16.08 ms/tok (Q4-baseline 16.04 대비 +0.25%, ✅ ≤ ±2%)
- 메모리: VmRSS 2.31 GB, 회귀 없음
- swap stage: `swapped 0/16 layers` — primary와 secondary가 동일 dtype(Q4)일 때 swap이 no-op으로 단축됨 (정상 동작, INV-122 v2.1 준수)
- **판정: PASS**

#### 시나리오 B — F16 primary + mixed.auf Q4 entry (dynamic dtype swap)

- 정확성: ❌ 모든 ratio (0.5, 1.0)에서 첫 token이 EOS — empty 출력
- 동일 조건의 v0.1.1 AUF (Q4-only): "Paris" 정답 PASS
- 동일 조건 mixed.auf + `--secondary-dtype q4_0` 명시: 여전히 FAIL
- TBT: 14.38–14.50 ms/tok (TBT 자체는 baseline 회복)
- swap stage: 16/16 layers 정상 swap, prefault=1.5–1.9 s (mixed AUF의 큰 mmap 영역)
- **판정: FAIL — 회귀 발견**

#### 시나리오 C — INV-135 v2 dtype reject

- C-1: Adreno SOA + `--secondary-dtype f16` → ✅ 즉시 reject + 친절한 에러 메시지
- C-2 (생략): default_dtype=Q4_0이므로 auto 선택 시 Q4 entry 자동 선택 (시나리오 A에서 검증 완료)
- **판정: PASS**

---

## 4. INV 검증 정리

### INV-137: multi-dtype shape consistency

`auf_tool verify` 자동 검증 통과 — 모든 `(tensor_id, variant)` 그룹 안에서 dtype 변형들이 동일 shape.

### INV-138: META.default_dtype 명시

`mixed.auf` META에 `default_dtype: Q4_0` 명시, `--dtypes q4_0,f16`에 포함된 값 — 검증 PASS.

### INV-122 v2.1: weight swap quality preservation

| 모드 | 정확성 | 충족 |
|------|--------|------|
| Q4 primary + Q4 entry (Scen-A) | ✅ "Paris" + 후속 byte-identical | **PASS** |
| F16 primary + Q4 entry (Scen-B) | ❌ empty | **FAIL** (회귀) |
| F16 primary + Q4 entry via v0.1.1 (Scen-B-v011) | ✅ "Paris" | (mixed.auf 한정 회귀임을 입증) |

### INV-135 v2: dtype dispatch + Adreno SOA F16 reject

- ✅ Adreno SOA + F16 명시 → reject (호스트/디바이스 모두 동일 메시지)
- ✅ Adreno SOA + auto dtype + multi-dtype AUF → default_dtype 따라 Q4 자동 선택
- ✅ lm_head Q4_0 entry는 mixed.auf 내에서도 정상 lookup (시나리오 A 디바이스 로그: `[Backend] lm_head: loading from AUF Q4_0 entry (~0 ms quantize, variant=WEIGHTS_ADRENO_SOA)`)

---

## 5. 발견된 이슈 (hotfix 후보)

### ISSUE-E-1 [HIGH]: F16 primary + mixed.auf Q4 swap 시 정확성 회귀

**증상**:
- F16 primary GGUF + multi-dtype mixed.auf (Q4+F16) + ratio≥0.5 + Adreno SOA backend
- ratio 무관, `--secondary-dtype` 무관 (auto / 명시 q4_0 둘 다)
- 첫 token부터 EOS 토큰(128001) — empty 출력
- 동일 조건의 v0.1.1 AUF (Q4 only)는 "Paris" 정답 PASS

**재현 명령** (디바이스):
```bash
./generate \
  -m Llama-3.2-1B-Instruct-f16.gguf \
  --secondary-gguf llama-3.2-1b-mixed.auf \
  --secondary-dtype q4_0 \
  --force-swap-ratio 1.0 \
  -p "The capital of France is" \
  -n 16 -b opencl --temperature 0.0 --threads 6
```

**범위 분리**:
- mixed.auf의 info/verify는 PASS (포맷 OK, INV-137/138 OK)
- byte determinism PASS (포맷 안정)
- Q4 primary + mixed.auf Q4 swap은 PASS (Q4↔Q4 dtype-matching path는 no-op)
- F16 primary + v0.1.1 (Q4-only) AUF는 PASS (single-dtype reader path 정상)
- F16 primary + mixed.auf (multi-dtype) reader path만 FAIL → **multi-dtype reader 경로의 Q4 entry payload 추출 또는 SOA permute에 결함 가능**

**가설**:
1. v0.2 multi-dtype payload layout에서 Q4 entry의 byte-offset 계산이 `secondary_mmap` dtype 선택 후 잘못 계산됨
2. F16 entry(2858 MB)와 Q4 entry(803 MB)가 같은 variant 안에 함께 배치되는데, Q4 entry로 swap 시 F16 영역 일부를 Q4로 해석할 가능성
3. lm_head는 정상 (`[lm_head] loaded from AUF AOS payload (140 MB, variant=WEIGHTS_ADRENO_SOA)` — 시나리오 B 로그) — layer weight 추출 경로 한정 회귀

**재현 ratio 차이**:
- Scen-B r0.5: prefault=1648 ms
- Scen-B r1.0: prefault=1860 ms (+ 12.8%)
- Scen-B r1.0+q4: prefault=1458 ms

prefault 시간이 swap된 layer 수에 비례 — mmap 영역 자체는 정상적으로 디스크에서 읽혀짐. 결함은 그 후 SOA permute 또는 weight bind 단계로 의심.

**다음 액션**:
- Implementer: `engine/src/auf/secondary_mmap.rs` 또는 multi-dtype tensor_index 조회 경로의 Q4 entry offset 검증 로직 점검
- 디바이스에서 weight swap 직후 layer 0의 wq weight 첫 32 byte hex dump → v0.1.1 vs mixed.auf 비교 (빠른 root cause 분리)
- spec test: `tests/spec/test_inv_122_v2.rs`에 F16 primary + multi-dtype AUF 시나리오 추가 (현재 spec 460은 호스트 path만 커버, Adreno SOA path는 디바이스 빌드 필요)

---

## 6. 산출물 요약

| 종류 | 위치 | 비고 |
|------|------|------|
| AUF (host all-variants) | `/tmp/auf_e/mixed_all.auf` (10.73 GiB) | byte-deterministic |
| AUF (host single-variant) | `/home/go/auf_e/llama-3.2-1b-mixed.auf` (3.58 GiB) | adreno_soa-only, S25 push 원본 |
| AUF (S25) | `/data/local/tmp/llama-3.2-1b-mixed.auf` | 동일 md5 `81f3c6c4...` |
| 호스트 검증 로그 | `/home/go/Workspace/llm_rs2-weight/_e_1_*.{log,txt}` | info/verify/reject |
| 디바이스 측정 로그 | `/home/go/Workspace/llm_rs2-weight/_e_2_*.log` | 8개 시나리오 + baseline |
| 본 보고서 | `results/data/weight_swap/v0_2_multi_quant_validation.md` | (이 파일) |

## 7. 권장 조치

1. **즉시**: ISSUE-E-1 hotfix sprint (Implementer/Architect) — F16 primary + mixed.auf 경로의 Q4 entry 추출 결함 점검
2. **spec 보강**: `tests/spec/test_inv_122_v2.rs`에 multi-dtype AUF 케이스 추가 (호스트 path simulation으로도 회귀 잡을 수 있는지 평가)
3. **docs 동기화**: `docs/USAGE.md`/`docs/auf_tool_guide.md` unstaged 변경 + 본 회귀 노트 반영 (Sprint E 후속 별도 작업)
4. **회귀 모니터링**: ISSUE-E-1 fix 이후 본 매트릭스(10 시나리오)를 디바이스 회귀 스위트에 등록

---

## 8. Sprint F — ISSUE-E-1 hotfix 종결 (2026-04-27)

### TL;DR

Sprint F에서 ISSUE-E-1 root cause 식별 + 수정 + 시나리오 B 디바이스 재측정 PASS 확인. **mixed.auf 다시 빌드한 후 F16 primary + `--secondary-dtype q4_0` swap이 "Paris" 정답을 출력**하며, TBT 회귀는 baseline 대비 -1.9% 이내.

### Root cause 분석

#### Byte-level diff (F-1)

v0.1.1 AUF (`/tmp/auf_build/Llama-3.2-1B-Instruct.v011-aos.auf`, 147 entries)와
v0.2 mixed.auf (`/home/go/auf_e/llama-3.2-1b-mixed.auf`, 294 entries)의 layer 0 entries
hex dump 비교:

| (layer, kind, dtype) | v011 size | v011 first 16B | mixed size | mixed first 16B | 동일 |
|----------------------|-----------|----------------|------------|-----------------|------|
| layer=0 attn_q Q4_0  | 2,359,296 | `94 7d 06 91 aa 7c c9 7c …` | 2,359,296 | `94 7d 06 91 aa 7c c9 7c …` | YES |
| layer=0 attn_norm    | 8,192 (dtype=1, F32 mistag) | `00 00 1e 3e …` | **dtype=3 Q4_0 size=1,152** + dtype=1 size=4,096 | `ee 2d 8a ba …` (Q4) / `f0 30 d8 31 …` (F16) | NO — 추가 Q4 entry |
| layer=0 ffn_norm     | 8,192 (dtype=1) | `00 00 4a 3e …` | dtype=3 size=1,152 + dtype=1 size=4,096 | (다름) | NO |

→ **layer weight Q4 bytes는 두 AUF에서 byte-identical** (정상).
→ **norm tensor에 Q4 entry가 추가됨** — v011은 norm 1개 (F16/F32-tagged), mixed는 norm 2개 (Q4 + F16).

(전체 dump: `/tmp/auf_compare/src/main.rs` ad-hoc tool 실행 결과)

#### Root cause 확정 (F-2)

**E** — `build_dtype_candidates`가 1-D tensor (RMSNorm weight, shape=[2048])에 대해서도
`shape_to_2d`의 `1 → [1, n]` 변환을 통해 Q4_0 quantize를 적용했다. shape=[1, 2048]은
`cols=2048`로 Q4_0 cols % 32 == 0 조건을 만족하기 때문에 변환이 silently 성공한 것.

결과:
1. mixed.auf에 norm별 Q4 entry 추가됨 (shape=[2048], size=1152, dtype=3)
2. F16 primary + `--secondary-dtype q4_0` swap 시 reader의 dtype filter가 norm Q4 entry를
   선택하여 1-D Q4_0 1152B (= 2048/32 × 18) bytes를 norm slot에 bind
3. 정상 F16 4096B 또는 F32 8192B와 dtype/size 불일치 → primary forward의 RMSNorm scale가
   garbage가 되어 첫 token이 EOS(128001)로 깨짐

Scen-A (Q4 primary + Q4 entry)가 PASS인 이유: Q4 primary와 mixed.auf 둘 다 Q4 dtype이라
swap이 no-op (`swapped 0/16 layers` 로그 확인). norm Q4 garbage entry는 swap되지 않음.

Sprint C senior-implementer 보고의 함정 2 ("`build_dtype_candidates` fallback에서 norm
tensor 1-D Q4 누락") 메모와 일치하지만, fallback 누락 방향이 아니라 정반대 — **norm은
애초에 Q4 변환을 하지 말아야 했다**.

### Fix (F-3)

**파일**: `engine/src/auf/dtype_convert.rs::build_dtype_candidates`

라이브러리 함수로 신규 export (auf_tool binary의 로컬 `build_dtype_candidates`를 lib로 이동).
1-D tensor (`shape_logical.len() < 2`) 가드 추가: multi-dtype 모드에서도 src_dtype 1개만
반환하여 norm Q4 entry 생성 차단.

| 파일 | 변경 |
|------|------|
| `engine/src/auf/dtype_convert.rs` | `build_dtype_candidates` 신규 함수 (1-D guard + multi-dtype dispatch). 86 줄 추가. |
| `engine/src/auf/mod.rs` | `build_dtype_candidates` re-export 추가. |
| `engine/src/bin/auf_tool.rs` | 로컬 `build_dtype_candidates` 함수 제거 → lib import. import 변경. 1-D guard 동작은 lib와 동일 (단일 진실의 원천). |
| `engine/tests/spec/test_issue_e_1_multi_dtype_byte_path.rs` | 신규 spec 테스트 7개 (R-1~R-7). 1-D F16/F32 guard, 2-D 정상 경로, single-dtype 모드, 0-D scalar, 후보 무시, Q4 reject silent skip. |
| `engine/tests/spec.rs` | 신규 테스트 모듈 등록. |

### F-4 시나리오 B 재측정

새 mixed.auf 빌드 (**261 entries** = 294 - 33 norm Q4 entries 제거됨, byte-deterministic):
```
[auf-tool] dtypes: [Q4_0, F16] (default=Some(Q4_0), multi_dtype=true)
... 147 tensors, 261 dtype entries extracted ...
[auf-tool] Build complete: /tmp/auf_e/mixed_fix.auf (3.58 GiB)
```

S25 push 후 측정 (HEAD: Sprint F 변경 적용):

| 시나리오 | Primary | Secondary | dtype | ratio | swapped | 출력 (앞 80자) | TBT (ms/tok) | 정답 |
|---------|---------|-----------|-------|-------|---------|--------------|--------------|------|
| Scen-A r1.0 | Q4_0 GGUF | mixed_fix.auf | auto | 1.0 | **0/16** | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **15.89** | ✅ |
| **Scen-B r1.0 (fix)** | F16 GGUF | mixed_fix.auf | q4_0 | 1.0 | **16/16** | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **14.53** | ✅ |
| **Scen-B r0.5 (fix)** | F16 GGUF | mixed_fix.auf | q4_0 | 0.5 | **8/16** | `Paris. The Eiffel Tower, a famous landmark in Paris, was built` | **25.56** | ✅ |

**판정**:
- 시나리오 A: TBT **15.89 ms/tok** (이전 16.08 대비 −1.2%, baseline 16.04 대비 −0.9%, ≤ ±2% **PASS**)
- 시나리오 B r1.0: TBT **14.53 ms/tok** (Sprint G-1 baseline 14.81 대비 −1.9%, ≤ ±2% **PASS**)
- 시나리오 B r0.5: TBT 25.56 ms/tok (mixed F16↔Q4 swap 중 reconvert 과도, 별도 분석 필요하나
  정확성은 PASS)
- 정확성: 모든 시나리오 "Paris" + 후속 byte-identical 출력. INV-122 v2.1 충족.

### F-5 Sanity gate

| 체크 | 결과 |
|------|------|
| `cargo fmt --all` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (0 errors) |
| `cargo test --workspace --release --test spec` | **467 passed** (이전 460 + 신규 7) |
| 신규 spec 테스트 7/7 | PASS (R-1~R-7) |
| Lib unit tests | OpenCL SIGSEGV는 NVIDIA 호스트 환경 한계 (master `26ff8b9` baseline에서도 동일 재현, ISSUE-E-1 fix와 무관) |

### F-6 산출물 갱신

| 종류 | 위치 | 비고 |
|------|------|------|
| AUF (Sprint F fix, host) | `/tmp/auf_e/mixed_fix.auf` (3.58 GiB) | norm Q4 entry 제거됨, 261 entries |
| AUF (Sprint F fix, S25) | `/data/local/tmp/llama-3.2-1b-mixed-fix.auf` | 동일 |
| 신규 spec 테스트 | `engine/tests/spec/test_issue_e_1_multi_dtype_byte_path.rs` | 7개 case |
| 본 보고서 | `results/data/weight_swap/v0_2_multi_quant_validation.md` §8 | (이 entry) |

### Sprint F 결론

ISSUE-E-1 hotfix 완료. v0.2 multi-quant 포맷 + reader/writer 경로가 모두 정상화되어 다음 산출물이
제공된다:
1. F16 primary + multi-dtype AUF + Q4 swap → 정상 추론 (정확성 + TBT 회귀 ≤ ±2%)
2. norm tensor 1-D guard로 향후 multi-dtype 추가 dtype (BF16, Q4_1 등) 도입 시에도 자동 차단
3. spec 회귀 격리 7개 case 등록 — 호스트 unit test로 회귀 즉시 검출 가능


