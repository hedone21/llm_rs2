# Sprint B-5b Phase 3 A — hybrid_attention 결정 라운드

**작성**: 2026-05-23
**Sprint**: B-5b Phase 3 A (Phase 0 후속 라운드)
**선행**: `arch/sprint_b5b_phase0_decision.md` + `.agent/todos/handoff_b5b_phase2_complete_2026_05_23.md` (HEAD `28bd7724`)
**다음 진입**: Phase 3 A 구현 (implementer)

---

## 본 라운드 목적

Phase 0 시점의 §J zone marker 적용안이 hot path 호출(`plan.rs:1553/1559`)의 부수효과(AtomicI32 store + Mutex lock + cl_mem 참조)와 §J 본문 "정책 query 한정 / read-only" 제약과 mismatch임이 메인 세션 review에서 확인됨. 사용자 결정 (verbatim): "A trait inversion으로 진행하자. ISP 위반이 누적되어 있고 이것은 리펙토링 전 스프린트 완료 후 다시 점검하자".

그러나 단순 Backend trait method 추가는 `engine/src/backend.rs`(L2)가 `crate::layers::hybrid_attention::*`(L3)을 import하게 되어 **L2→L3 INV-LAYER-002 신규 위반**을 만든다. 본 라운드에서 4개 결정으로 이 충돌을 해소하고 Phase 3 A를 단일 단계로 종결한다.

---

## R-P3A-1. hybrid_attention 모듈 위치 = (γ) 부분 격상 (free function + thread-local + Setup struct만 L2)

**선택**: hybrid_attention 모듈의 *식별자 일부*를 L2로 격상한다. 격상 대상은 다음 5개로 한정.

| 식별자 | 종류 | 현재 위치 | 격상 후 위치 | 격상 근거 |
|---|---|---|---|---|
| `compute_kv_split` | free fn | `layers/hybrid_attention.rs` | `engine/src/hybrid_attention.rs` (L2) | 순수 함수, OpenCL feature 무관 |
| `current` | free fn | `layers/hybrid_attention.rs` | 동상 | thread-local read, OpenCL feature 무관 |
| `install` | free fn | `layers/hybrid_attention.rs` | 동상 | thread-local write, OpenCL feature 무관 |
| `HybridScope` | RAII struct | `layers/hybrid_attention.rs` | 동상 | thread-local guard, OpenCL feature 무관 |
| `HybridAttnSetup` | struct (OpenCL feature gated) | `layers/hybrid_attention.rs` | 동상 | 양 도메인(L1 backend + L3 inference + L4 session) 대등 사용 |
| `HybridGpuBuffer` | struct (OpenCL feature gated) | `layers/hybrid_attention.rs` | 동상 | `HybridAttnSetup` 필드. 분리 시 §G 정합성 깨짐 |
| `ENV_KV_FRAC` | const | `layers/hybrid_attention.rs` | 동상 | env flag 식별자, 양 도메인 공유 |

**모듈 전체를 L2로 옮긴다** (현 파일 그대로 이동, 컴파일 가드는 그대로 유지).

### 옵션 검토

**(α) L3 유지 + Backend trait이 L3 import**: L2→L3 INV-LAYER-002 신규 위반 1건. 본질 미해소 + 정책 위치 시프트. **기각**.

**(β) §G 모듈 통째 격상**: §G 본문 "data identifier" 정의에서 약간 벗어남 (free function + thread-local + RAII guard가 섞임). 다만 *모듈이 도메인 어휘 자산화*되었다는 점에서 §G 정신과 일치 — `partition_workspace.rs`/`kv_cache_ops.rs`/`op_kind.rs`/`cpu_kernels.rs`/`secondary.rs`가 이미 L2에 동급으로 존재하며 `engine/src/lib.rs:11~22`에 등록됨. 다만 §G 본문 조건 검증 필요.

**(γ) 부분 격상**: (β)와 동일하나 정신적 강조 — 격상 단위를 *모듈 한 단위*로 보고, 그 안의 식별자가 §G 본문 "data identifier" 조건 일부에만 합치한다는 점을 명시.

→ **(γ) 채택**. 격상 단위는 *모듈 전체*이되 §G 본문 "data identifier" 조건은 `HybridAttnSetup`/`HybridGpuBuffer`/`HybridScope`/`ENV_KV_FRAC`가 직접 만족. free function 3개(`compute_kv_split`/`current`/`install`)는 *해당 data identifier에 부수되는 access API*로 함께 격상한다는 정당화.

### §G 허용 조건 검사

| 조건 | 만족 여부 | 근거 |
|---|---|---|
| Type 종류 = data identifier | △ (struct/RAII는 만족, free fn 3건은 *부수 access API*로 동반 격상) | `HybridAttnSetup`/`HybridGpuBuffer`/`HybridScope`/`ENV_KV_FRAC` 4건 직접 만족. free fn 3건은 해당 struct들의 *생성·조회·소멸* API. §G 본문 "RAII guard 등은 본 정책 밖" 문구와 충돌하지만, RAII guard가 **본 격상의 핵심 객체 그 자체**가 아니라 *thread-local install API의 부산물*이라는 점에서 본 라운드 해석으로 정리한다. |
| 사용 분포 = 양 도메인 대등 | ○ | L1 backend (`plan.rs:1553/1559`), L3 inference (`transformer.rs:2300`), L4 session (`prologue.rs:95/429/520/653` 4건). 단방향이 아님 — install은 L4, current는 L1+L3 양쪽, compute_kv_split은 L1만 |
| 위치 정합성 | ○ | `engine/src/op_kind.rs`/`kv_cache_ops.rs`/`partition_workspace.rs`/`cpu_kernels.rs`/`secondary.rs`와 동급. OpenCL feature gating은 L2 traits에 이미 존재(예: `Backend::as_opencl_secondary` `#[cfg(feature = "opencl")]`) → 신규 패턴 아님 |

§G 본문 "RAII guard 등은 본 정책 밖" 문구는 *RAII guard만 단독 격상*하는 경우를 제외하려는 의도로 해석. 본 격상은 `HybridAttnSetup` 격상 + 그 lifetime을 관리하는 `HybridScope` 동반 격상이므로 격상 단위가 "data identifier + 그 access API 패밀리"이며 §G 정신과 부합한다. 본 해석을 §G 본문에 register 행 신설로 명시한다 (R-P3A-4 참조).

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| L2가 OpenCL feature 의존하는 struct(`HybridGpuBuffer`) 보유 | 낮음 | 이미 동일 패턴(`Backend::as_opencl_secondary`)이 L2에 존재. `#[cfg(feature = "opencl")]`가 식별자 단위로 적용되어 non-opencl 빌드에서는 식별자가 사라짐 |
| §G 본문 "RAII guard 본 정책 밖" 문구 위반 우려 | 중간 | R-P3A-4에서 §13.8-G **register 행 신설**로 본 격상의 해석을 명시. 신규 정책 신설 아니며 운용 메모 등급 |
| 호출자 4건 path 갱신 필요 (transformer.rs:2300, prologue.rs 4건, plan.rs:1553/1559) | 낮음 | 단순 mechanical rename. 호스트 cargo check + S25 microbench 없이 게이트 통과 가능 |

---

## R-P3A-2. Backend trait method 시그니처 = N/A (Backend trait 미수정)

**선택**: R-P3A-1 §G 격상으로 `crate::hybrid_attention::*` 자체가 L2 식별자가 되므로 **Backend trait에 신규 method를 추가하지 않는다**. plan.rs:1553/1559의 호출은 `crate::layers::hybrid_attention::*` → `crate::hybrid_attention::*`로 path 단순 갱신만 진행.

### 옵션 검토

**(a) `fn current_hybrid_setup(&self) -> Option<Arc<HybridAttnSetup>>`**: trait method가 thread-local read를 wrap하지만, thread-local 자체는 backend instance와 무관 → 모든 backend impl이 동일 helper 호출하여 의미 없는 vtable indirection. ISP 누적 +1. **기각**.

**(b) `fn with_hybrid_kv_split<R>(&self, kv_len, kv_frac, f: &dyn Fn(usize, &HybridAttnSetup) -> R) -> Option<R>`**: closure 시그니처 무거움 + Setup이 None일 때 panic 우회 로직이 closure 안으로 들어가 plan.rs 본문 가독성 ↓. ISP 누적 +1. **기각**.

**(c) `fn execute_hybrid_attn_split(&self, ...)`**: plan.rs:1531~1722 191 LOC 전체를 backend method로 흡수 — Phase 3 A scope에서 가능한 작업량이 아님. Phase 4+ regression risk. **기각**.

→ **R-P3A-1 §G 격상으로 R-P3A-2가 불요화됨**. ISP 누적 +0, Backend trait method 수 61 → 61 유지. 사용자 명시 "ISP 위반 누적 sprint 완료 후 재점검" 정책과 정합.

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| §G 격상 실패 시 Backend trait inversion fallback 필요 | 낮음 | Phase 3 A 구현 단계에서 §G 격상이 cargo check 실패하면 즉시 본 결정 재논의(architect 1회 라운드). 본 sprint scope는 §G 격상 단일 path |
| ISP 누적 영구 미해결 우려 | 중간 | 사용자 명시 backlog 항목 (handoff_b5b_phase2_complete §3 "Backend trait 분할 정책 재논의"). 본 라운드 결정으로 ISP 누적이 *증가하지 않음* (61 유지) |

---

## R-P3A-3. plan.rs 후속 struct field 접근 처리 = (i) plan.rs에서 직접 접근

**선택**: §G 격상으로 `HybridAttnSetup`/`HybridGpuBuffer`가 L2 type이 되므로, plan.rs(L1)에서 `setup.ready_flags_gpu.host_ptr() as *mut AtomicI32` + `setup.partial_ml_cpu.lock()` 직접 접근은 **L1 → L2 normal import**이며 INV-LAYER 위반 아님. plan.rs 본문은 import path 갱신(`crate::layers::hybrid_attention::*` → `crate::hybrid_attention::*`)만 진행.

### 옵션 검토

**(i) 직접 접근 유지**: §G 격상 후 정상 L1 → L2 의존. **채택**.

**(ii) backend method로 wrap (R-P3A-2 (c)와 결합)**: 작업량 폭증, Phase 4+ scope. **기각**.

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| pub field 4개 (`ready_flags_gpu`/`partial_ml_gpu`/`partial_o_gpu`/`partial_ml_cpu`/`partial_o_cpu`) + `HybridGpuBuffer::host_ptr()` pub method가 L1에서 직접 dereferenced | 낮음 | 현재 코드 동작과 등가. 본 sprint scope는 위치 이동만, 가시성/캡슐화 강화는 별도 backlog |
| L2의 `HybridAttnSetup`이 OpenCL ocl::Buffer<u8> 보유 → L2 OpenCL 의존 강화 | 낮음 | 이미 `backend.rs::GpuEvent`/`backend.rs::cl_mem` 의존 패턴 존재 (Backend trait L2 위치에서 OpenCL feature gated method 다수). 신규 결합 아님 |

---

## R-P3A-4. ARCHITECTURE.md §G register 행 신설 = §13.8-G에 hybrid_attention register 행 추가

**선택**: §13.8-G 본문 *수정 없이* §G 적용 register 행을 **별도 sub-list로 신설**한다. §13.8 정책 수는 5개(F/G/H/I/J) 유지 — 신규 정책 아님.

### 추가 위치

`ARCHITECTURE.md`의 §13.8-G 본문 끝 (현재 line ~1542 "5건 이상 누적 시 §13.4에 *promotion register* 표 신설 검토" 다음 줄)에 다음 register sub-list를 추가.

```markdown
- **§G 적용 register** (5건 미만이므로 sub-list 형식 유지):
  - `OpKind` (B-2a sprint, RESOLVED) — `observability/profile/op_trace.rs` → `engine/src/op_kind.rs`
  - `KVCacheOps` (B-5b Phase 1, RESOLVED) — `pressure/kv_cache.rs` trait 부분 → `engine/src/kv_cache_ops.rs`
  - `PartitionWsCell` / `PartitionWorkspace` (B-5a sprint, RESOLVED) — `layers/tensor_partition.rs` 일부 → `engine/src/partition_workspace.rs`
  - `CpuKernelSet` / `OpenClSecondary` / `SecondaryStore` (B-5b Phase 2, RESOLVED) — Backend trait capability 인프라
  - **`hybrid_attention` 모듈** (B-5b Phase 3 A, PLANNED) — `layers/hybrid_attention.rs` → `engine/src/hybrid_attention.rs`. 격상 단위는 모듈 전체 (struct/RAII/free fn 패밀리). §G 본문 "RAII guard 본 정책 밖" 문구는 *RAII guard 단독 격상*을 제외하려는 의도이며, 본 사례는 `HybridAttnSetup` data identifier + 그 lifetime을 관리하는 `HybridScope` 동반 격상 패턴으로 §G 정신과 부합.
```

### ARCHITECTURE.md §13.5 V-?? plan 행 갱신

기존 `V-?? (hybrid_attention)` plan 행 (line 1370)을 다음으로 교체:

```markdown
| **V-?? (hybrid_attention)** | TBD (B-5b sprint Phase 3 A) | hybrid_attention 모듈을 §13.8-G shared identifier promotion으로 L2 격상 (`engine/src/hybrid_attention.rs` 신규). plan.rs:1553/1559 + transformer.rs:2300 + prologue.rs 4건 호출자가 새 import path 사용. Backend trait method 추가 없음 (ISP 누적 +0). §J zone marker 부착은 폐기 — plan.rs hot path 호출이 §J 본문 "read-only 정책 query 한정" 제약과 mismatch (AtomicI32 store + Mutex lock + cl_mem 참조 부수효과 발생). baseline -2 (V-02 hybrid_attention 2건 해소). | 없음 |
```

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| §G register 행이 §13.8 정책 수 증가로 오인 | 낮음 | "신규 정책 아님 / sub-list 형식 유지 / 5건 미만이므로 표 신설 불요" 명시. §F의 운용 메모 패턴과 동일 |
| §J 본문은 *수정 금지* (B-5a sprint 결정 보존) | 낮음 | §J 본문 무수정, 단지 V-?? plan 행에서 §J 적용 폐기를 1줄로 기록. §J 정책 자체는 `build_partition_plan`에 그대로 유효 |
| §G 본문 "RAII guard 본 정책 밖" 문구 vs 본 격상의 모순 우려 | 중간 | register 행에 *해석 명시*. 본문 무수정. 본 해석이 본 sprint scope 한정이며 향후 RAII guard 단독 격상 요청 시에는 §G 본문 재해석 필요(별도 라운드) |

---

## Phase 분해 (Phase 3 A → 3 B → 3 C 분리)

| Phase | 작업 | 게이트 | baseline 변동 (절대) |
|---|---|---|---|
| 3 A (본 라운드 후속) | hybrid_attention 모듈 §G 격상 + 호출자 4건 path 갱신 + ARCHITECTURE.md §G register sub-list + V-?? plan 행 갱신 | 호스트 게이트 7개 + layer_lint 새 violation 0 + spec test 8 PASS | 208 → 206 (-2, V-02 hybrid_attention 2건 해소) |
| 3 B | as_opencl_secondary 치환 (qnn_oppkg downcast 제거) | 호스트 게이트 + S25 microbench Δ ≤ +3% + spec test 8 PASS | 206 → ?? (`OpenClSecondary` trait body 설계 결정 α/β/γ 후 결정. 사용자 명시 보류 상태) |
| 3 C | 잔여 cleanup (Phase 4 종결 commits에 흡수) — `SecondaryStore` placeholder 처분, `intra_token_yield_enabled()` dead code 처분, `gpu_yield_impl` 중복 흡수 검토 | 호스트 게이트 + layer_lint baseline regen | Phase 4 종결 시 결정 |

### Phase 3 A 단일화 정당화

handoff_b5b_phase2_complete의 §3 Phase 3 진입 직전 결정 3건 중:
- **Backend trait 분할 정책** — 사용자 명시 보류 (본 라운드 scope 외)
- **`OpenClSecondary` trait body 설계 (α/β/γ)** — Phase 3 B로 분리 (Backend trait 분할 정책 결정 후 진행)
- **hybrid_attention §J 확장 범위** — 본 라운드에서 §G로 결정 → Phase 3 A로 단일화

Phase 3 A는 *§G 격상 단일 path*이며 ISP 누적 +0이므로 Backend trait 분할 정책 결정과 독립적으로 진행 가능. Phase 3 B는 Backend trait 결정 후 별도 sprint 명시.

### S25 microbench 생략 정당화

Phase 3 A의 코드 변경은:
1. `engine/src/layers/hybrid_attention.rs` → `engine/src/hybrid_attention.rs` 파일 이동
2. `engine/src/lib.rs`에 `pub mod hybrid_attention;` 등록, `engine/src/layers/mod.rs`에서 제거
3. plan.rs:1553/1559 + transformer.rs:2300 + prologue.rs 4건 + (잠재적 4건 더) import path 갱신: `crate::layers::hybrid_attention::*` → `crate::hybrid_attention::*`

코드 logic 무변경. 컴파일러 monomorphization 결과 동일 (LLVM IR diff 0 예상). Phase 2 Stage 2-A 패턴(LOC 적은 mechanical 단계 → microbench 생략)과 동일 정당화.

---

## §13.8 정책 영향

- **§13.8 정책 수**: 5개(F/G/H/I/J) 유지. **신규 정책 신설 없음**.
- **§G register sub-list 신설**: 본문 무수정, 적용 사례 6건 누적 (5건 미만 → 표 형식 신설 불요, sub-list로 충분).
- **§J 본문**: 무수정. B-5a sprint 결정 보존. plan.rs hybrid_attention 호출지에 §J 적용은 폐기 (V-?? plan 행에서만 명시).
- **spec/41-invariants.md INV-LAYER-001 비고**: 무수정. Phase 1 결정 보존.

---

## Phase 3 A 위임 prompt 초안 (implementer용)

```
## 본 작업 = B-5b Phase 3 A (hybrid_attention §G 격상)

### 핵심 컨텍스트
- 선행: arch/sprint_b5b_phase3_round.md (R-P3A-1~4 결정 완료)
- handoff_b5b_phase2_complete_2026_05_23.md HEAD 28bd7724
- 본 라운드 직전 결정: §J zone marker 적용은 hot path 부수효과와 mismatch → §G 격상으로 전환

### 작업 (mechanical, ~30분)

1. **파일 이동**: `engine/src/layers/hybrid_attention.rs` → `engine/src/hybrid_attention.rs`
   - 파일 내용 무변경 (cfg gate 그대로, test 모듈 그대로)

2. **모듈 등록 갱신**:
   - `engine/src/lib.rs` line 11~22 사이에 `pub mod hybrid_attention;` 추가 (알파벳순 위치)
   - `engine/src/layers/mod.rs:2`의 `pub mod hybrid_attention;` 라인 삭제

3. **호출자 import path 갱신** (전체 search & replace):
   - `crate::layers::hybrid_attention::` → `crate::hybrid_attention::`
   - 영향 파일 (사전 확인 6건, 추가 발견 시 동일 패턴 적용):
     - engine/src/backend/opencl/plan.rs (line 1553, 1559)
     - engine/src/models/transformer.rs (line 2300)
     - engine/src/session/decode_fallback/prologue.rs (line 95, 429, 520, 653)
     - 기타 grep 결과의 모든 호출자

4. **layer_lint baseline regen**:
   - `python scripts/layer_lint.py --baseline > engine/tests/spec/inv_layer_baseline.json`
   - V-02 hybrid_attention 2건 (line 26/35) 제거 확인 (208 → 206)

5. **ARCHITECTURE.md 갱신**:
   - line 1370 V-?? (hybrid_attention) plan 행 → 본 라운드 R-P3A-4 명시 텍스트로 교체
   - §13.8-G 본문 끝(line ~1542)에 §G register sub-list 추가 (본 라운드 R-P3A-4 텍스트)
   - §J 본문 무수정

6. **commit**: `refactor(layer): B-5b Phase 3 A — hybrid_attention 모듈 §G L2 격상`

### 게이트 (S25 microbench 생략)

| Gate | 명령 | 기대 |
|---|---|---|
| cargo build | `cargo build --release -p llm_rs2` | OK |
| cargo fmt | `cargo fmt --all -- --check` | clean |
| cargo clippy | `cargo clippy --workspace -- -D warnings` | 0 warnings |
| cargo test --lib | `cargo test --lib` | 회귀 0 |
| spec inv_layer | `cargo test --test spec inv_layer` | 8 PASS |
| layer_lint | `python scripts/layer_lint.py --baseline` | violation 0 (baseline JSON regen 후) |

**S25 microbench 생략 정당화**: 코드 logic 무변경 (파일 이동 + import path rename 단일). monomorphization 결과 동일. Phase 2 Stage 2-A 패턴과 동일.

### 자기점검
- [ ] hybrid_attention.rs 파일 이동 후 컴파일 OK
- [ ] V-02 hybrid_attention 2건 baseline JSON에서 제거됨 확인
- [ ] §G register sub-list ARCHITECTURE.md에 추가됨
- [ ] §J 본문 무수정 확인
- [ ] spec/41-invariants.md INV-LAYER-001 본문 무수정 확인
- [ ] notify-send 알림
```

---

## 자기점검

- [x] **결정 4건 모두 단일안** (R-P3A-1 γ / R-P3A-2 N/A / R-P3A-3 i / R-P3A-4 register 행 신설)
- [x] **각 결정의 risk + 버린 옵션 근거** 명시 (R-P3A-1 α/β 기각, R-P3A-2 a/b/c 기각, R-P3A-3 ii 기각)
- [x] **baseline 변동 수치** 명시: 208 → 206 (-2, V-02 hybrid_attention 2건 해소). 현재 baseline JSON 카운트는 handoff에 208 명시. Phase 0 plan의 216 → 186 목표는 본 라운드에서 208 → 206로 갱신 (Phase 1/2 누적 -8 결과 반영)
- [x] **정책 인플레이션 회피**: §13.8 정책 수 5개(F/G/H/I/J) 유지. §G register sub-list는 신규 정책 아님 (운용 메모 등급, §F 패턴과 동일)
- [x] **§J 본문 무수정 / spec INV-LAYER-001 비고 무수정** 명시
- [x] **§G "RAII guard 본 정책 밖" 문구 vs 본 격상 모순 해석** 명시 (register 행에서 *RAII guard 단독 격상 제외 의도*로 해석)
- [x] **Phase 3 A 단일화** (3 B는 Backend trait 분할 정책 결정 후 별도, 3 C는 Phase 4 흡수)
- [x] **S25 microbench 생략 정당화** 명시 (Stage 2-A 패턴 인용)
- [x] **implementer prompt 초안** 본문 포함 (구현·게이트·자기점검 6요소)
- [x] **본 라운드 자체는 코드 무수정** — arch/spec/ARCHITECTURE.md 문서만 변경
