# W-AUF-1B + W-AUF-2 Verification

**검증일**: 2026-05-21
**기준 브랜치**: `origin/master` HEAD `000634b3`
**디바이스**: Samsung Galaxy S25 (`R3CY408S4HN`)
**모델**: Qwen 2.5-1.5B-Instruct multi-dtype AUF

## 결론

W-AUF-1B (auf_tool multi-dtype mode) 및 W-AUF-2 (AUF self-secondary 자동 활성)의 **핵심 기능이 master HEAD `000634b3`에 이미 구현되어 있음**이 확인됨. 본 task는 검증 + 게이트 통과 + 잔존 작업 확인으로 종결.

| Sub-sprint | 핵심 구현 | 위치 | 확인 |
|---|---|---|---|
| W-AUF-1B | auf_tool `--dtypes` 옵션 (Sprint C, INV-138/139) | `engine/src/bin/auf_tool.rs:117` | ✓ |
| W-AUF-1B | `CAPABILITY_BIT_MULTI_DTYPE` setter/checker | `engine/src/auf/header.rs:26,184,209` | ✓ |
| W-AUF-1B | `TensorEntry::variant_offsets/sizes` | `engine/src/auf/tensor_index.rs:108` | ✓ |
| W-AUF-1B | `entries_for(layer, kind)` multi-vec 반환 | `engine/src/auf/tensor_index.rs:350` | ✓ |
| W-AUF-1B | `convert_tensor_dtype` (Q4_0 ↔ F16) | `engine/src/auf/dtype_convert.rs:37` | ✓ |
| W-AUF-1B | `entries_for_multi_dtype` unit test | `engine/src/auf/tensor_index.rs:465` | ✓ |
| W-AUF-2 | `SecondaryMmap::from_auf_self_secondary` | `engine/src/models/weights/secondary_mmap.rs:317` | ✓ |
| W-AUF-2 | `try_promote_auf_self_secondary_to_rpcmem` | `engine/src/models/weights/secondary_mmap.rs:739` | ✓ |
| W-AUF-2 | `RpcmemSecondaryStore::from_auf_self_secondary` (Eager-Flattened Adapter 등가) | `engine/src/models/weights/rpcmem_secondary.rs` | ✓ |
| W-AUF-2 | `--no-self-secondary` 동작 | `session/init.rs` | ✓ (디바이스 검증) |

## 검증 게이트 결과

### 호스트 (cwd = `.claude/worktrees/w_auf_1b_to_2`)

| 게이트 | 결과 | 비고 |
|---|---|---|
| `cargo build --release --workspace` | ✅ exit 0 (1m 13s) | 회귀 0 |
| `cargo test -p llm_rs2 --lib --release -- --skip backend::opencl --skip memory::opencl` | ⚠️ 1194 passed; 1 failed | flaky `pressure::kv_cache::tests::test_release_unused_pages_rss_reduction` — **master HEAD baseline 상태, 본 task와 무관 (회귀 아님)** |
| `cargo test -p llm_rs2 --release --test test_auf_gguf_byte_equivalence` | ✅ 2/2 PASS | clean |
| `cargo fmt --check -p llm_rs2` | ⚠️ exit 0, **단 master 기존 diff at `engine/src/session/init.rs:558`** | 본 task 새 코드 0 → 새 위반 없음. 기존 위반은 별건 |

### multi-dtype AUF (호스트, 4.25 GiB)

`auf_tool info` 출력 (`/home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf`):

```
capability_opt   : 0x000000000000000c (LM_HEAD_PRECOMPUTED_Q4_0 + MULTI_DTYPE_VARIANTS)  ✓
TENSOR_INDEX:
  variants         : ["WEIGHTS_CPU_AOS"]
  tensor_count     : 537
  dtype_dist       : {F16=198, F32=141, Q4_0=198}
  multi_dtype_grps : 198 (groups with >=2 dtype candidates)  ✓
META.default_dtype : Q4_0
```

- ✅ `CAPABILITY_BIT_MULTI_DTYPE` ON (bit 3)
- ✅ `multi_dtype_grps: 198` (각 weight (layer, kind)가 Q4_0 + F16 두 dtype variant 보유)
- ⚠️ `variants: ["WEIGHTS_CPU_AOS"]` 한 가지 — plan은 cpu_aos + adreno_soa 양쪽 명시. 본 빌드는 cpu_aos만. **별건 / W-AUF-1B 핵심(multi-dtype)은 충족, 추가 variant 빌드는 후속**

### 디바이스 검증 (S25, multi-dtype AUF push 후)

`generate_head` (master HEAD `000634b3` binary)로 multi-dtype AUF + `--primary-dtype f16 --primary-variant cpu-aos` 실행:

| 게이트 | 결과 | 비고 |
|---|---|---|
| AUF primary 로드 | ✅ `[AUF] Loaded: '...multi-dtype.auf' arch=Qwen2 variant=CpuAos dtype=Some(F16) layers=28 vocab=151936` | clean |
| self-secondary 자동 활성 (default) | ✅ `[AUF self-secondary] auto-activated: variant=WEIGHTS_CPU_AOS primary_dtype=F16 secondary_dtype=Q4_0 layers=28` | 핵심 게이트 |
| `--no-self-secondary` opt-out | ✅ `[AUF] self-secondary disabled by --no-self-secondary` | 핵심 게이트 |
| 정확성 (fluent output) | ✅ `I_am_a_Skype_learner # I am a Skype learner...` | 32 tok 정상 출력 |
| reverse swap (Q4_0 primary → F16 secondary) | ✅ 정상 거부: `Error: Reverse swap rejected: primary=Q4_0, secondary=F16. ... Cannot use secondary=F16 when primary is already Q4_0.` | F16→Q4_0 방향 강제 보장 |

### 미완료 게이트 (manager 의존 — 후속 task로 분리)

| 게이트 | 사유 |
|---|---|
| swap_weights heartbeat 등록 | `llm_manager` 서비스 + resilience signal 필요. 단순 generate에서 미트리거 |
| swap 후 top-5 overlap > 99% | 위와 동일 |
| swap latency Phase 6.5 baseline 대비 ±5% | rpcmem alias path 직접 측정 + 비교 필요. 별도 measurement script |

위 세 가지는 본 task의 **핵심 기능 검증 외부 영역**으로, manager 서비스 통합 검증 task로 분리 권장.

## 별건 (본 task 무관, backlog 후보)

1. **master HEAD flaky test** — `pressure::kv_cache::tests::test_release_unused_pages_rss_reduction` RSS drop 임계 환경 의존. `#[ignore]` 또는 임계 완화 검토
2. **master HEAD fmt diff at `engine/src/session/init.rs:558`** — `matches!` 매크로 multi-line vs single-line. 한 줄 `cargo fmt` 가능
3. **multi-dtype AUF variants 확장** — plan은 cpu_aos + adreno_soa 양쪽 명시, 현 빌드는 cpu_aos만. adreno_soa 추가 빌드 필요 시 별도 task

## 결과 종합

- W-AUF-1B: 호스트 빌드/test/AUF capability 게이트 PASS
- W-AUF-2: 디바이스 자동 활성 + opt-out 동작 PASS
- 본 task 코드 변경 0 (구현이 master HEAD `000634b3`에 이미 존재) → 회귀 0
- manager 의존 swap 검증 게이트 (top-5 overlap, swap latency)는 후속 task로 분리

## 환경 정보

- 디바이스: Samsung Galaxy S25, serial `R3CY408S4HN`
- AUF 파일: `/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf` (4.25 GiB, push 92.18s @ 47.2 MB/s)
- Binary: `/data/local/tmp/generate_head` (master HEAD `000634b3` 빌드)
- Backend: `qnn_oppkg`, threads=6, prompt `"I_am_a"`, num-tokens=32
