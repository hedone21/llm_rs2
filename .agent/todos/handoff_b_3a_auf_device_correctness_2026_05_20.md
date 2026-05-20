# Handoff — B-3a AUF primary device 정확성 회복 (진행 중, 2026-05-20)

## 진입 문장

**"B-3a 진행"** — AUF primary path가 device에서 garbage output을 내는 이슈를
byte-level diff로 격리.

## 현재 상태

- **Worktree**: `.claude/worktrees/sprint1_auf_loader` (브랜치 `worktree-sprint1_auf_loader`)
- **HEAD**: `ad41e3d4 feat(loader): AUF tied 모델 lm_head 추론 fix (B-3a 진행 중)`
- **호스트 게이트**: `cargo test -p llm_rs2 --lib --no-default-features` **1159 PASS, 회귀 0**
- **W-AUF-1/1B/2 호스트 작업 완료**. self-secondary 자동 활성, R8 trait, lm_head tied fix 모두 commit됨

## 증상 — Garbage output (S25 디바이스)

3-way control 측정 (qnn_oppkg, 6T, qwen2.5-1.5b):

| 구성 | Decode (ms/tok) | Avg TBT (ms) | 정확성 |
|---|---|---|---|
| GGUF primary (q4_0.gguf) | 3.51 | 29.24 | ✅ "I am a high school senior..." |
| AUF primary q4_0.auf (ADRENO_SOA variant) | 7.55 | 25.63 | ❌ "Parameter(Parameter(..." 반복 |
| AUF primary q4_0-aos.auf (CPU_AOS variant) | 9.28 | 33.31 | ❌ "公 hart reap早晚妇女..." |
| AUF primary multi-dtype CPU_AOS F16 + self-secondary | 3.84 | 50.04 | ❌ 다른 garbage |
| AUF CPU_AOS + OpenCL backend (qnn_oppkg 대신) | 22.72 | 27.43 | ❌ garbage — backend 무관 |

→ **backend 무관, variant 무관**으로 AUF primary path 자체가 깨짐. GGUF는 정상이라
   AUF 빌드/로드의 어딘가에 byte mismatch 또는 의미적 불일치.

## 이미 시도된 fix (commit 적용됨)

### 1. AUF default_tag CpuAos 변경 (`0ec863c3`)
- variant_select.rs::default_tag() opencl feature → BackendTag::CpuAos
- AufSource::open에 Auto fallback chain 추가 (CpuAos → AdrenoSoa → CudaAos)
- 사용자 결정 — Adreno도 AOS primary 기본. SOA는 명시 시에만.
- 결과: 동작 변경됐지만 garbage 지속

### 2. AUF norm/cross-layer lenient lookup (`69fd5991`)
- `is_dtype_strict_kind(id)` — weight matmul만 dtype strict, 나머지는 None (META.default_dtype + first-match)
- 결과: `--primary-dtype f16` + multi-dtype AUF에서 AttnNorm F32 lookup 통과. garbage 패턴은 변경됐지만 여전히 garbage.

### 3. tied 모델 lm_head 추론 fix (`ad41e3d4`)
- 배경: Sprint G-1-B에서 AUF가 tied 모델도 LmHead Q4_0 entry를 precompute하므로 기존 `find_lm_head_entry().is_none()` 추론이 항상 false.
- `infer_tie_word_embeddings(view)`: token_embd.weight shape == LmHead shape면 tied 추론
- `has_tensor(LmHead)`가 tied 시 false → load_model이 `lm_head = embed.clone()` GGUF path와 일치
- 결과: garbage 지속. lm_head는 1차 원인 아님.

## 이미 검증된 가설 (원인 아님)

| 가설 | 검증 | 결과 |
|---|---|---|
| ADRENO_SOA layout이 forward에 잘못 해석 | CPU_AOS variant도 garbage | 원인 아님 |
| CPU_AOS 64-byte padding | 텐서 크기 모두 64로 나뉨 (padding 0) | 원인 아님 |
| Qwen2의 unpermute_qk_rows 차이 | 둘 다 `arch == Llama` 한정, Qwen2는 unpermute skip | 정합 |
| Shape outermost-first vs innermost-first | GGUF rev → outermost = AUF shape 그대로 → 같은 결과 (정사각 + non-정사각 모두) | 정합 |
| Build script q4_0 → q4_0 변환 | `build_dtype_candidates`가 `dt == src_dtype`이면 `src_bytes.to_vec()` 그대로 | 정합 |
| tied lm_head 추론 미스매치 | 위 fix 3 적용 | 원인 아님 |
| `needs_dequant_fallback`(Q4_0) | 둘 다 false → zero-copy mmap buffer | 정합 |
| eos_token_id 차이 | 둘 다 u32::MAX fallback (forward 정확성 무관) | 정합 |
| qnn_oppkg backend 특수 처리 | OpenCL backend도 garbage | 원인 아님 |

## 남은 가설 (다음 세션 진단 대상)

1. **AUF byte payload가 GGUF와 byte-level 미일치** — 가장 유력. build_variant_tensor_bytes 또는 build_dtype_candidates 어딘가에 미발견된 차이가 있을 수 있음.
2. **Shape 차이 (특히 non-정사각 텐서)** — wk/wv (`[256, 1536]`)에서 outermost-first 해석이 다른 결과를 낼 가능성. 위 분석에서 "정합"으로 판단했지만 실제 forward indexing을 재확인 필요.
3. **AufViewBuffer의 ptr alignment** — AUF mmap 내부에서 텐서 시작 ptr가 GPU buffer alignment를 만족 안 할 가능성. GGUF는 page-aligned mmap, AUF는 weights section 시작이 64KB align되지만 텐서 자체는 그 안에서 cursor 배치.
4. **forward path의 dtype 분기** — token_embd 또는 다른 cross-layer 텐서가 GGUF에서는 F32→F16 변환 (load_f32_as_f16) 적용되는데 AUF에서는 F16 그대로. 미세한 layout 차이?

## 다음 진단 단계 (옵션 A — byte-level diff, 추천)

### Step 1: Spec test 작성

`engine/tests/spec/test_auf_gguf_byte_equivalence.rs` (NEW) 신설:

```rust
#[test]
fn auf_primary_tensor_bytes_match_gguf_tied_model() {
    // 1. GGUF source 열기
    let gguf_path = "models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf";
    let gguf = GgufSource::open(Path::new(gguf_path)).unwrap();

    // 2. AUF source 열기 (CpuAos variant)
    let auf_path = "models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf";
    let auf = AufSource::open(Path::new(auf_path), AufVariantChoice::CpuAos,
                              AufDtypeChoice::Auto, None).unwrap();

    // 3. ModelConfig 차이 dump
    assert_eq!(gguf.config().hidden_size, auf.config().hidden_size);
    // ... 모든 필드 dump

    // 4. 같은 텐서 (예: blk.0.attn_q.weight) bytes 비교
    let mem = Galloc::new();
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let gguf_t = gguf.load_tensor_cpu(
        &TensorId::LayerWeight { layer: 0, kind: LayerWeightKind::Wq },
        true, &mem).unwrap();
    let auf_t = auf.load_tensor_cpu(
        &TensorId::LayerWeight { layer: 0, kind: LayerWeightKind::Wq },
        true, &mem).unwrap();

    // 5. shape / dtype / bytes 모두 비교
    assert_eq!(gguf_t.shape().dims(), auf_t.shape().dims(), "shape mismatch");
    assert_eq!(gguf_t.buffer().dtype(), auf_t.buffer().dtype(), "dtype mismatch");
    let gguf_bytes = unsafe {
        std::slice::from_raw_parts(gguf_t.buffer().as_ptr(), gguf_t.buffer().size())
    };
    let auf_bytes = unsafe {
        std::slice::from_raw_parts(auf_t.buffer().as_ptr(), auf_t.buffer().size())
    };
    assert_eq!(gguf_bytes.len(), auf_bytes.len(), "byte len mismatch");

    // first mismatch position 출력
    for i in 0..gguf_bytes.len() {
        if gguf_bytes[i] != auf_bytes[i] {
            panic!("byte diff at offset {}: gguf={:02x} auf={:02x}", i, gguf_bytes[i], auf_bytes[i]);
        }
    }
}
```

### Step 2: 비교 텐서 확장
- blk.0.attn_q.weight (위)
- blk.0.attn_norm.weight (F32 norm)
- token_embd.weight (F16 embed)
- blk.0.ffn_gate.weight (Q4_0 non-정사각? 실은 [8960, 1536] 비대칭)
- 첫 diff가 발견되는 텐서 + offset이 핵심 단서

### Step 3: diff 결과로 좁히기
- shape 차이 → AufSource의 shape 처리 fix
- dtype 차이 → build_dtype_candidates 또는 ggml_type_to_tensor_dtype 매핑 fix
- bytes 차이 → AUF builder에 적용된 변환 추적 (Q/K permute, F32→F16 변환 등)

### Step 4: device 재측정
- spec test fix 후 device generate으로 GGUF↔AUF byte-equivalent + 출력 정합 확인.

## 검증 인프라

- 호스트 GGUF: `models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf` (1.13 GiB, ggml type Q4_0)
- 호스트 AUF (CPU_AOS): `models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf` (1.25 GiB)
- 호스트 AUF (ADRENO_SOA): `models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.auf` (1.25 GiB)
- 호스트 AUF (multi-dtype): `models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf` (4.24 GiB)
- 디바이스 push 완료 (S25 `/data/local/tmp/models/qwen2.5-1.5b/`)
- 디바이스 바이너리 `/data/local/tmp/generate` (HEAD `ad41e3d4` 빌드)
- 디바이스 libs: `/data/local/tmp/qnn/libQnn*.so` + `/data/local/tmp/libqnn_oppkg.so` + `/data/local/tmp/libcdsprpc.so`

## 관련 파일 위치 (HEAD `ad41e3d4`)

| 항목 | 경로 |
|---|---|
| AufSource primary | `engine/src/models/loader/auf/source.rs` |
| AUF builder | `engine/src/bin/auf_tool.rs` (extract_weight_blobs, build_dtype_candidates, build_variant_tensor_bytes) |
| dtype convert | `engine/src/auf/dtype_convert.rs` |
| GGUF source | `engine/src/models/loader/gguf.rs` (load_raw, unpermute_qk_rows, qk_permute_shape) |
| ModelConfig::from_gguf_metadata | `engine/src/models/config.rs:228~` |
| from_auf_meta | `engine/src/models/loader/auf/source.rs:242` |
| tie 추론 (방금 fix) | `engine/src/models/loader/auf/source.rs:285~298` |
| AUF reader / weights_bytes | `engine/src/auf/reader.rs:99~102` |

## 환경 / 규칙

- 언어: 한국어
- EnterWorktree 필요 없음 (이미 worktree 안)
- 빌드: `python scripts/run_device.py -d galaxy_s25 generate` (NDK env + qnn feature 자동)
- 호스트: `cargo build/test -p llm_rs2 --lib`
- S25 6T 고정 (`--threads 6`)
- TBT metric은 항상 avg_tbt
- `.cl` 커널 수정은 본 sprint 범위 아님
- third_party / libs는 master 디렉토리 symlink (`/home/go/Workspace/llm_rs2/{third_party,libs}`)

## Plan / 종결 handoff

- W-AUF-2 종결: `.agent/todos/handoff_sprint1_w_auf_2_complete_2026_05_20.md`
- W-AUF-1 종결: `.agent/todos/handoff_sprint1_auf_loader_complete_2026_05_20.md`
- Plan: `/home/go/.claude/plans/proud-strolling-whale.md`

## 재진입

**"B-3a 진행"** — Step 1 spec test 작성부터 시작.
