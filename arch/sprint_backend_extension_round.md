# Sprint Backend Trait Extension — Architect 라운드

**작성**: 2026-05-24
**Sprint**: Backend trait extension (B-5b 후속, 절충안 sprint)
**선행**: `.agent/todos/handoff_b5b_complete_2026_05_23.md` (HEAD `eba122b2`) · `arch/sprint_b5b_phase3_round.md` · `arch/sprint_b5b_phase0_decision.md`
**다음 진입**: 사용자 라운드 (escalate 3건) → Phase 1 architect finalize → Phase 2 implementer

---

## 본 라운드 목적

사용자가 절충안 확정 (hot path 보존 + cold path만 string lookup) + researcher 조사 (llama.cpp `ggml-backend` `get_proc_address(name)` + ExecuTorch `BackendInterface`) 반영. R2 게이트(KIVI 3건 hot path 제외) 후 Phase 3 scope = **cold path 4건 + cleanup**으로 축소. 본 라운드에서 3 결정 + Phase 분해 + ARCHITECTURE.md 갱신 항목 + escalate 3건 확정.

### 본 sprint scope 명시 (Phase 3 최종)

| # | 호출지 | path 분류 | 현재 코드 |
|---|---|---|---|
| 1 | `backend/qnn_oppkg/mod.rs:139` (`with_opencl_secondary` 클로저 내부) | cold (swap path) | `be.as_any().downcast_ref::<OpenCLBackend>()` |
| 2 | `models/weights/secondary_mmap.rs:752` (`backend_supports_rpcmem_secondary` 후속) | cold (weight loading) | `backend.as_any().downcast_ref::<QnnOppkgBackend>()` |
| 3 | `models/weights/secondary_mmap.rs:793` (`try_open_rpcmem_secondary`) | cold (weight loading) | 동일 |
| 4 | `models/transformer.rs:1057` (`convert_weights_to_q4` SOA 진입) | cold (loader path) | `backend.as_any().downcast_ref::<OpenCLBackend>()` |

**제외**: `pressure/kivi_cache.rs:1559/1842/2108` — per-token × 16 layer hot path. 본 sprint scope 외 (이미 R2 게이트로 확정).

**partition_workspace 2건 (149/175)**: 사용자 컨텍스트에서 Phase 3 scope로 명시되었으나 실측 결과 *backend* downcast가 아닌 *Buffer trait → UnifiedBuffer* downcast (`cpu_merge_staging.buffer().as_any().downcast_ref::<UnifiedBuffer>()`). 본 sprint Backend trait extension scope와 무관 → **별도 backlog** ("Buffer trait extension"). escalate 질문 Q3 참조.

→ **확정 scope**: 4건 (qnn_oppkg 1건 + secondary_mmap 2건 + transformer 1건). 모두 OpenCL 또는 QNN downcast.

---

## R-EXT-1. Extension key namespace = (α) 모듈 const string 평면 namespace

**선택**: `engine/src/backend.rs`에 `pub const EXT_OPENCL_QUEUE: &str = "opencl_queue";` / `pub const EXT_OPENCL_SECONDARY: &str = "opencl_secondary";` / `pub const EXT_QNN_RUNTIME: &str = "qnn_runtime";` 정의. Backend trait에 `fn get_extension(&self, name: &str) -> Option<&dyn std::any::Any> { None }` default impl 추가. 각 backend가 자신이 노출할 extension만 override.

### 옵션 trade-off

| 옵션 | 장점 | 단점 | RPN |
|---|---|---|---|
| **(α) const string** | ggml 직역 / 단순 / 새 backend 추가 시 trait sig 무변경 (OCP) / cold path만 사용해 string compare 비용 무관 | 오타 silent None / namespace 충돌 잠재 | 90 |
| (β) enum-keyed `BackendExt::OpenClQueue` | 컴파일타임 검증 / exhaustive match | 새 extension 추가 시 enum 변경 = trait sig 변경 = OCP 약화 / 새 backend가 외부 crate에서 자기 extension key를 정의할 수 없음 / Rust enum이 trait sig에 박힘 | 140 |
| (γ) typed accessor per ext (`fn opencl_queue(&self) -> Option<&OpenCLBackend>`) | type-safe / downcast 불필요 | trait method N개 추가 = ISP 누적 영구화 (B-5b sprint가 풀려는 문제 자체) / researcher 권장 패턴(string lookup)과 모순 | 180 |

→ **(α) 채택**. ggml `get_proc_address(name)` 패턴 직역으로 사용자 의도 그대로. researcher 조사 결과와 정합. silent None risk는 R-EXT-3 anti-pattern guard로 완화 (rustdoc + `// COLD-EXT` 마커 + cold path 컨벤션).

### 인터페이스 시그니처

```rust
// engine/src/backend.rs (트레잇 본문에 추가, hot path와 격리된 cold-ext 섹션)

/// Cold-path extension namespace. Backend-specific capabilities (queue handle,
/// rpcmem fns, secondary slot 등) 을 string key 로 노출한다. Hot path (per-op /
/// per-layer / per-token) 호출 금지 — vtable + downcast + HashMap lookup 가
/// 누적된다. 사용처는 swap path / loader path / init path 만 허용.
///
/// 호출 컨벤션:
/// - `name` 은 본 모듈 `EXT_*` 상수만 사용.
/// - 반환 `&dyn Any` 는 호출자가 `downcast_ref::<ConcreteType>()` 로 추출.
/// - 미지원 backend 는 `None` 반환 (default impl).
fn get_extension(&self, _name: &str) -> Option<&dyn std::any::Any> {
    None
}
```

```rust
// 모듈 상수 (Backend trait 위, namespace 단일화)
pub const EXT_OPENCL_QUEUE: &str = "opencl_queue";         // -> &OpenCLBackend
pub const EXT_OPENCL_SECONDARY: &str = "opencl_secondary"; // -> &OpenCLBackend (fallback slot)
pub const EXT_QNN_RUNTIME: &str = "qnn_runtime";           // -> &QnnOppkgBackend
```

`get_extension` 반환은 `&self` 또는 self 내부 field. lifetime 은 `&self` 와 동일하므로 `&'_ dyn Any`. Arc clone 불요 (cold path 단일 호출).

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| 오타로 silent `None` 반환 (e.g. `"opencl_queu"`) → cold path 실패가 silent | 중간 (RPN 150 = 5×6×5) | (1) 호출지가 반드시 `EXT_*` 상수만 사용. grep 정책: `get_extension("`를 lint 검사 (Phase 4 호스트 게이트로 추가). (2) 모든 호출지가 `Option::ok_or_else(...)` 또는 `?` 로 명시적 에러 전파. silent panic 없음. (3) Phase 2 호스트 테스트에 "각 backend가 자기 extension key 모두 반환" 단위 테스트 1건 추가. |
| Namespace 충돌 (외부 crate가 같은 key 정의) | 낮음 (RPN 80) | 본 engine crate 한정. 외부 crate plugin 시나리오 없음. 본 sprint scope에서 무관. |
| `&dyn Any` 반환이 type erasure로 downcast 비용 발생 | 낮음 (RPN 60) | cold path 한정 (per-swap / per-load / per-init) → ms 단위 누적 무관. KIVI hot path는 본 sprint scope 외. |
| ggml처럼 함수 포인터 반환 (`Option<fn()>`) 가 아닌 `&dyn Any` 객체 반환으로 ggml 패턴과 미세 차이 | 낮음 (RPN 40) | `get_proc_address` 의 본질은 "name lookup → opaque value"이며, Rust 의 `&dyn Any` 가 동등한 추상화. C 함수 포인터 ≠ Rust trait object 의 자연스러운 차이. 본질 동일. |

---

## R-EXT-2. `OpenClSecondary` placeholder + `SecondaryStore` 처분 = (a) 본 sprint에서 제거 + Phase 5 흡수

**선택**:
- `engine/src/secondary.rs` 의 `SecondaryStore` trait 를 본 sprint Phase 5 (cleanup) 에서 *제거*한다.
- B-5b Phase 2 에서 도입되어 사용처가 0건이며, 본 sprint Phase 3 의 cold path 4건이 `get_extension(EXT_OPENCL_SECONDARY)` 또는 `get_extension(EXT_QNN_RUNTIME)` 으로 흡수되므로 `SecondaryStore` 추상화 의도가 *cold path extension 으로 변용* 되었다.
- `OpenClSecondary` placeholder (B-5b Phase 2 Stage 1 도입, 이미 `eba122b2` 에서 본체 제거됨) 의 잔재 (Backend trait 의 `as_opencl_secondary` method 1개) 는 Phase 5 에서 제거. `get_extension(EXT_OPENCL_SECONDARY)` 로 대체.

### 옵션 trade-off

| 옵션 | 장점 | 단점 | RPN |
|---|---|---|---|
| **(a) 본 sprint Phase 5 에서 제거** | 사용처 0 / extension 패턴이 더 일반적 / ISP 누적 -1 (`as_opencl_secondary` method 제거) | B-5b Stage 1 인프라 도입 의도 일부 폐기 (실제 implementation 없음 → 폐기 비용 = 문서 + 1 trait 제거만) | 60 |
| (b) 유지 + 본 sprint 에서 `SecondaryMmap`/`RpcmemLayerRegion` impl 적용 (B-5b Stage 1 본 의도 실현) | Stage 1 의도 완성 | (1) cold path 4건이 *extension lookup* 패턴과 *trait dispatch* 패턴으로 분기 → 일관성 깨짐. (2) `SecondaryStore` 의 `as_bytes()/len()` 가 cold path 4건의 *실제 사용 패턴* 과 mismatch (4건 모두 *backend 자체*를 추출하지 *bytes* 를 추출하지 않음). (3) 작업량 +1~2 commit | 130 |
| (c) 유지하되 본 sprint scope 에서 손대지 않음 | scope creep 회피 | 영구 dead code 잔존 / Phase 5 cleanup 약속만 남아 미래 sprint debt | 90 |

→ **(a) 채택**. `SecondaryStore` 의 *byte access* 추상화는 본 sprint cold path 4건의 실제 사용 패턴(backend handle 추출)과 mismatch. 사용처 0 + B-5b 인프라 도입 commit (`eb1970dc`) 폐기 비용이 미미 (문서 1개 + trait 1개) → ROI 가장 높음. `as_opencl_secondary` Backend method (B-5b Stage 1 도입, +1 ISP 누적) 도 `get_extension(EXT_OPENCL_SECONDARY)` 로 대체되어 영구 제거 → **ISP 누적 -1 (61 → 60)**.

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| B-5b Phase 2 Stage 1 commit (`eb1970dc`) 폐기 = sprint 작업 일부 무효화 | 낮음 (RPN 60) | Stage 1 의도(`OpenClSecondary` capability trait)가 실측 cold path 패턴과 mismatch 였음을 인정. *retrospective correction* 으로 본 라운드에 명시. handoff `b5b_complete` Landmines #6 에 이미 "placeholder 처분 필요" 명시 — 본 결정으로 종결. |
| `cuMemHostRegister` byte span 추상화 (V-09 `SecondaryMmapBytes`) 와 혼동 | 낮음 (RPN 40) | `SecondaryMmapBytes` (engine/src/memory/secondary.rs) 는 *별개 trait* 으로 잔존. 본 sprint 무관. 본 sprint 가 제거하는 것은 `engine/src/secondary.rs::SecondaryStore` 뿐. |
| §G 적용 register sub-list (`ARCHITECTURE.md` §13.8-G) 에 `SecondaryStore` 가 RESOLVED 로 등재됨 → 제거 시 register 갱신 필요 | 낮음 (RPN 30) | Phase 5 에서 register sub-list 의 `SecondaryStore` 항목 제거 (1줄 삭제). §13.8-G 정책 본문 무변경. |

---

## R-EXT-3. Phase 3 치환 시 anti-pattern guard = (ⅰ+ⅲ) rustdoc + `// COLD-EXT` 마커

**선택**: `Backend::get_extension` 본문의 rustdoc 에 *hot path 호출 금지* 를 명시 + 호출지 4건 모두에 `// COLD-EXT: <swap/loader/init>` 마커 주석 부착. clippy 강제는 본 sprint scope 외 (RPN 미만).

### 옵션 trade-off

| 옵션 | 장점 | 단점 | RPN |
|---|---|---|---|
| **(ⅰ+ⅲ) rustdoc + `// COLD-EXT` 마커** | 인적 검증 + grep 가능 / 호스트 게이트에 grep policy 추가 가능 / 작업량 minimal | 강제력 없음 / 신규 호출지가 마커 누락 가능 | 80 |
| (ⅱ) 별도 `ColdExt: Backend` sub-trait 분리 | 컴파일타임 강제 / typestate 효과 | (1) 모든 호출지가 `&dyn ColdExt` cast 필요. (2) Backend ↔ ColdExt 양방향 trait object cast 가 Rust 에서 trivial 하지 않음 (`Arc<dyn Backend>` → `Arc<dyn ColdExt>` 변환 boilerplate). (3) 새 backend 가 ColdExt 도 impl 해야 함 → ISP 누적 +1 (본 sprint 가 풀려는 문제). | 160 |
| (ⅳ) clippy custom lint (`disallowed-methods`) | 자동 강제 | 구현 비용 큼 (clippy custom lint plugin 또는 build.rs hack) / 본 sprint scope 초과 | 200 |

→ **(ⅰ+ⅲ) 채택**. 본 sprint scope = 4건 (`qnn_oppkg/mod.rs:139` + `secondary_mmap.rs:752/793` + `transformer.rs:1057`) 으로 인적 검증 가능. 신규 호출지가 안 생기는 한 충분. 추후 5건 이상 누적 또는 hot path 의심 호출지 발견 시 (ⅱ) 또는 (ⅳ) 재검토 (backlog).

### Guard 구체화

1. **rustdoc 본문** (R-EXT-1 시그니처 부분 참조):
   - "Hot path (per-op / per-layer / per-token) 호출 금지" 1줄 명시.
   - "사용처는 swap path / loader path / init path 만 허용" 명시.

2. **호출지 마커 주석** (4건 모두 부착):
   ```rust
   // COLD-EXT: swap path (qnn_oppkg with_opencl_secondary fallback)
   let ocl = be.get_extension(EXT_OPENCL_QUEUE)?
       .downcast_ref::<OpenCLBackend>()?;
   ```
   마커 prefix `// COLD-EXT:` + path 분류 (swap / loader / init).

3. **호스트 게이트에 grep policy 추가** (Phase 4):
   ```bash
   # 호스트 게이트 추가 항목
   # cold-ext call site 전수조사 — 마커 누락 검출
   ! grep -rn 'get_extension(' engine/src/ \
       | grep -v '// COLD-EXT' \
       | grep -v 'engine/src/backend.rs' \
       | grep -v 'engine/tests/'
   ```
   호출지가 마커 없으면 grep 결과 비어있지 않음 → CI fail.

### Risk + 완화

| Risk | 심각도 | 완화 |
|---|---|---|
| 신규 호출지가 마커 누락 (인적 lapse) | 중간 (RPN 100) | Phase 4 호스트 게이트 grep policy 가 검출. PR review 단계 차단. |
| Marker 가 잘못된 path 분류 (e.g. `// COLD-EXT: hot path`) | 낮음 (RPN 40) | Marker 본문에 "swap / loader / init" 외 값 거부 (grep policy 확장 가능). 본 sprint Phase 4 grep policy 1차 버전은 단순 존재 검사만. |
| Path 분류가 시간이 흐르며 변질 (cold → warm/hot) | 중간 (RPN 90) | 본질적으로 사회적 합의. 별도 backlog "cold/warm/hot path 정의 명시화" 등록 — 본 sprint 무관. |
| 마커 fragmentation: `// COLD-EXT` vs `// COLD-EXT:` vs `// cold-ext` | 낮음 (RPN 50) | Phase 4 grep policy 가 단일 패턴(`// COLD-EXT:` prefix)으로 강제. 단일 source-of-truth 는 본 라운드 문서. |

---

## Phase 분해

| Phase | 작업 | 게이트 | baseline 변동 (절대) |
|---|---|---|---|
| 1 | 본 라운드 finalize (사용자 escalate 3건 후) + ARCHITECTURE.md §13.5 V-?? Resolution Log 새 entry plan + §13.8 정책 영향 분석 | architect 결정 3건 단일안 + escalate 응답 반영 | 196 (B-5b 종결 시점, 무변동) |
| 2 | `Backend::get_extension` 본체 추가 + 3 `EXT_*` 상수 + `OpenCLBackend`/`QnnOppkgBackend` override + 호스트 단위 테스트 ("각 backend 가 자기 extension key 모두 반환") | cargo build + clippy 0 + spec test 8 PASS + 새 단위 테스트 PASS + S25 microbench 생략 (cold-path 추가, hot-path 무변경) | 196 (인프라 추가만) |
| 3 | 4 호출지 치환 (downcast → `get_extension`) + `// COLD-EXT:` 마커 + 호출지 grep 검증 | cargo build + clippy 0 + spec test 8 PASS + 호스트 게이트 grep policy 단순 검증 | 196 → 192 (-4, 4건 downcast 해소) |
| 4 | 호스트 게이트 grep policy `scripts/` 추가 (`scripts/check_cold_ext.sh` 또는 기존 `scripts/layer_lint.py` 확장) + 호스트 게이트 CI integration | grep policy PASS + 신규 violation 0 | 192 (정책만) |
| 5 | cleanup: `engine/src/secondary.rs::SecondaryStore` 제거 + `Backend::as_opencl_secondary` method 제거 (B-5b Phase 2 +1 ISP 누적분 회수) + §13.8-G register sub-list 갱신 (`SecondaryStore` 행 제거) + ARCHITECTURE.md §13.5 V-?? RESOLVED entry 마무리 | cargo build + clippy 0 + spec test 8 PASS + layer_lint baseline regen | 192 → ~192 (구조적 -1 ISP, baseline 절대 변동 0 또는 minor) |
| 6 | 통합 handoff + master FF 결정 | handoff R1~R6 + 자기점검 | — |

### S25 microbench 생략 정당화 (Phase 2~5)

- **Phase 2**: `get_extension` default impl `None` 반환 + override (CPU/CUDA 는 `None` 유지) 만 추가. Hot path 호출 0건 → vtable cost 0.
- **Phase 3**: 4 호출지가 모두 cold path (swap / loader / init). Per-token / per-layer / per-op 호출 0 — 측정 nuisance 이하.
- **Phase 5**: `SecondaryStore` trait + `as_opencl_secondary` method 제거는 dead code 제거. monomorphization 결과 동일.

B-5b Phase 2 Stage 2-A 패턴 (Δ −0.231%) 인용. 다만 *호스트 게이트 합산 wall-clock* 은 Phase 3 commit 후 1회 측정 후 (관행적으로) handoff 에 부록 형태로 첨부 권장 (사용자 결정).

### ISP 누적 변동

| 시점 | Backend trait method 수 | 변동 |
|---|---|---|
| B-5b 진입 (`fe1e8e19`) | 61 | — |
| 본 sprint Phase 2 | 62 (+1 `get_extension`) | +1 |
| 본 sprint Phase 5 | 61 (`as_opencl_secondary` 제거) | -1 |
| **본 sprint 종결** | **61 (net 0)** | |

`get_extension` 1개로 cold-path capability N개를 흡수 → ISP 누적이 영구 +1 (N) 가 아닌 영구 +0 (1 method + namespace 확장). 새 cold-path capability 추가 시 string 상수 1개만 추가하면 됨 (OCP 강화).

---

## ARCHITECTURE.md 갱신 항목

### §13.5 Resolution Log 신규 entry

기존 `V-?? (gpu_yield)` / `V-?? (KVCacheOps)` / `V-?? (hybrid_attention)` 패턴 (line 1367~1370) 에 새 행 추가:

```markdown
| **V-?? (cold_ext_namespace)** | TBD (Backend extension sprint) | Backend trait `get_extension(name: &str) -> Option<&dyn Any>` default impl + `EXT_OPENCL_QUEUE` / `EXT_OPENCL_SECONDARY` / `EXT_QNN_RUNTIME` 모듈 상수 도입. cold path 4건 downcast 치환 (`qnn_oppkg/mod.rs:139` + `secondary_mmap.rs:752/793` + `transformer.rs:1057`). B-5b Phase 2 Stage 1 의 `OpenClSecondary` capability trait + Backend `as_opencl_secondary` method 는 본 sprint Phase 5 에서 제거 (실측 cold path 패턴이 byte access 가 아닌 backend handle 추출 → mismatch). ISP 누적 net 0 (62→61). baseline -4 (4건 downcast 해소). hot path (`pressure/kivi_cache.rs:1559/1842/2108`) 는 본 sprint scope 외 (R2 게이트). | KIVI 3건 hot path |
```

### §13.8 새 정책 라인 ─ 본 결정: §13.8-K 신설 *불요*

본 sprint 는 ggml 의 `get_proc_address` 패턴 차용으로 기존 §13.8 정책 (§F~§J) 의 *변경* 없이 `Backend::get_extension` 단일 method 로 흡수된다. 정책 인플레이션 회피 우선 (B-5b sprint 결정 보존).

**대신 §13.8-G register sub-list 갱신**:

- `OpenClSecondary` / `SecondaryStore` 행 (B-5b Phase 2 Stage 1 RESOLVED) 을 **REVOKED (Backend extension sprint Phase 5, mismatch with cold path pattern)** 으로 변경 또는 제거.
- `CpuKernelSet` (B-5b Phase 2 Stage 1, RESOLVED) 는 그대로 유지 — cold path mismatch 아님, NEON kernel function pointer set 으로 의도대로 활용 중.

### §13.8-G 본문 영향

§13.8-G "Shared identifier promotion" 정책 자체는 무변경. `SecondaryStore` 의 §G 적용이 *실측 후 mismatch 로 revoke* 된 사례로 register 에 명시 (운용 메모 등급 — §G 정책 자체의 변경 아님).

### §13.4 directory migration map 영향

`engine/src/secondary.rs` 행 (있을 경우) 제거. Phase 5 cleanup 에서 일괄 처리.

---

## §13.8 정책 영향 요약

- **§13.8 정책 수**: 5개(F/G/H/I/J) 유지. **신규 정책 신설 없음**.
- **§13.8-G register sub-list**: `OpenClSecondary` / `SecondaryStore` 행 REVOKED 표기 + 본 sprint 명시. `CpuKernelSet` 유지.
- **§13.8-J zone marker**: 본 sprint 적용 없음 (B-5a sprint 결정 보존).
- **spec/41-invariants.md INV-LAYER-001 비고**: 무수정. B-5b Phase 1 결정 보존.
- **새 §13.8-K 신설 후보 ("Cold-path extension namespace")**: **기각**. 본질이 *Backend trait 의 1 method 추가* 로 흡수되어 별도 정책 신설 가치 < 정책 인플레이션 비용.

---

## 추가 escalate 질문 (사용자 라운드 필요)

### Q1. `with_opencl_secondary` closure API 폐기 vs 유지

`qnn_oppkg/mod.rs:128~142` 의 `pub fn with_opencl_secondary<R>(&self, f: impl FnOnce(&OpenCLBackend) -> R) -> Option<R>` closure API 는 *backend 내부*에서 secondary slot 접근을 캡슐화한다. Phase 3 치환 후 두 가지 path:

- **path A (단순 치환)**: closure body 안의 `be.as_any().downcast_ref::<OpenCLBackend>()` 만 `be.get_extension(EXT_OPENCL_QUEUE).and_then(|a| a.downcast_ref::<OpenCLBackend>())` 로 치환. closure API 그대로 유지. **변경 minimal**.
- **path B (closure 폐기)**: 호출자 (Phase 3 의 4건 중 qnn_oppkg 호출자) 가 직접 `qnn_backend.get_extension(EXT_OPENCL_QUEUE)` 를 호출하고 closure 패턴 폐기. `with_opencl_secondary` 함수 자체 제거. **변경 light → medium** (호출자 1건 확인 필요).

→ Architect 추천: **path A**. 본 sprint 의 trait extension 본질과 closure API 폐기는 독립 결정. closure 패턴 폐기는 별도 backlog.

질문: A / B 중 선택?

### Q2. `engine/src/secondary.rs` 모듈 처분

`SecondaryStore` 제거 후 모듈 본문 비어버림 (29 lines → 0). 두 path:

- **path X**: 모듈 파일 자체 제거 (`engine/src/lib.rs` 의 `pub mod secondary;` 도 삭제). `engine/src/memory/secondary.rs::SecondaryMmapBytes` 는 별개 모듈로 잔존.
- **path Y**: 모듈 파일 유지 + 빈 docstring 만 남김 (미래 cold-ext 관련 type 통합 위치 예약).

→ Architect 추천: **path X**. 빈 모듈 유지는 dead code 신호. 사용처 0 + 의도가 실측 mismatch 로 revoke 된 상황에서 모듈 잔존 가치 < cleanup ROI.

질문: X / Y 중 선택?

### Q3. partition_workspace.rs 2건 (149/175) Backend extension 무관 처리 확인

사용자 컨텍스트의 Phase 3 scope 표에 `partition_workspace.rs:149,175` 가 포함되어 있으나, 실측 결과 **Backend downcast 가 아닌 Buffer trait → UnifiedBuffer downcast** (`cpu_merge_staging.buffer().as_any().downcast_ref::<UnifiedBuffer>()`).

- 본 sprint Backend trait extension 으로 흡수 불가 (Buffer trait 의 extension 이 필요한 별개 issue).
- 별도 backlog "Buffer trait extension" 또는 §J zone marker 검토 대상.

→ Architect 처리: **본 sprint scope 에서 분리 + 별도 backlog 등록**.

질문: 본 분리 확인. 별도 backlog 우선순위 (P1 / P2 / P3)?

---

## Phase 2~5 위임 prompt 초안 (implementer 용)

```
## 본 작업 = Backend trait extension sprint Phase 2~5

### 핵심 컨텍스트
- 선행: arch/sprint_backend_extension_round.md (R-EXT-1~3 결정 완료, escalate Q1~Q3 사용자 단일안 반영)
- handoff_b5b_complete_2026_05_23.md HEAD eba122b2
- 본 라운드 결정: §13.8 무변경 / get_extension 단일 method 흡수 / cold-path 4건 + cleanup

### Phase 2 (인프라)
1. engine/src/backend.rs 에 EXT_OPENCL_QUEUE / EXT_OPENCL_SECONDARY / EXT_QNN_RUNTIME 상수 + Backend::get_extension default impl 추가
2. OpenCLBackend override (engine/src/backend/opencl/mod.rs):
   - get_extension(EXT_OPENCL_QUEUE) → Some(self as &dyn Any)
   - get_extension(EXT_OPENCL_SECONDARY) → Some(self as &dyn Any) (호환성)
3. QnnOppkgBackend override (engine/src/backend/qnn_oppkg/mod.rs):
   - get_extension(EXT_QNN_RUNTIME) → Some(self as &dyn Any)
4. 호스트 단위 테스트: 각 backend 가 자기 extension key 를 반환 + 미지원 key 는 None
5. cargo build + clippy 0 + spec test 8 PASS

### Phase 3 (4건 치환)
1. backend/qnn_oppkg/mod.rs:139 — // COLD-EXT: swap path
2. models/weights/secondary_mmap.rs:752 — // COLD-EXT: loader path (AUF self-secondary)
3. models/weights/secondary_mmap.rs:793 — // COLD-EXT: loader path (try_open_rpcmem_secondary)
4. models/transformer.rs:1057 — // COLD-EXT: loader path (convert_weights_to_q4 SOA)
5. 호스트 게이트 grep: `grep -rn 'get_extension(' engine/src/ | grep -v '// COLD-EXT' | grep -v 'engine/src/backend.rs'` 가 비어있어야 함
6. layer_lint baseline regen — 4건 downcast 제거 확인 (196 → 192)

### Phase 4 (정책 강제)
1. scripts/check_cold_ext.sh (또는 layer_lint.py 확장) 신설 — grep policy 자동화
2. 호스트 게이트 CI 항목 추가
3. spec test 새 entry (선택) — `engine/tests/spec/inv_layer_cold_ext.rs` 에 grep 게이트 1건

### Phase 5 (cleanup)
1. engine/src/secondary.rs 파일 제거 (escalate Q2 답이 path X 일 때)
2. engine/src/lib.rs 의 `pub mod secondary;` 제거
3. Backend::as_opencl_secondary method 제거 (B-5b Phase 2 Stage 1 도입)
4. OpenCLBackend / 5 backend 의 as_opencl_secondary override 제거
5. ARCHITECTURE.md §13.8-G register sub-list 의 OpenClSecondary / SecondaryStore 행 → REVOKED 표기
6. ARCHITECTURE.md §13.5 V-?? (cold_ext_namespace) RESOLVED 갱신
7. layer_lint baseline regen 최종

### 게이트
- cargo build + clippy 0 + spec test 8 PASS — 각 Phase
- layer_lint baseline 새 violation 0 (Phase 3 후 -4, Phase 5 후 net 0~-1)
- 호스트 게이트 grep policy PASS (Phase 4 후)
- S25 microbench 생략 정당화 (cold path only, hot path 무변경)

### 자기점검
- [ ] get_extension default impl None 확인
- [ ] 3 backend (CPU/OpenCL/QNN) 모두 build 통과
- [ ] 4 호출지 모두 // COLD-EXT: 마커 부착
- [ ] layer_lint baseline 4건 감소
- [ ] §13.8-G register sub-list REVOKED 표기
- [ ] notify-send 알림
```

---

## 자기점검 (architect 라운드)

- [x] **결정 3건 모두 단일안** (R-EXT-1 α / R-EXT-2 a / R-EXT-3 ⅰ+ⅲ)
- [x] **각 결정의 옵션 trade-off 표 + RPN ≥ 100 리스크 대응** 명시
- [x] **R-EXT-1**: 옵션 β/γ 기각 (β RPN 140 = OCP 약화, γ RPN 180 = ISP 누적 본질 문제)
- [x] **R-EXT-2**: 옵션 b/c 기각 (b 사용 패턴 mismatch RPN 130, c 영구 dead code RPN 90)
- [x] **R-EXT-3**: 옵션 ⅱ/ⅳ 기각 (ⅱ ISP 누적 +1 RPN 160, ⅳ 구현 비용 RPN 200)
- [x] **Phase 분해 + 게이트 명령** (Phase 1~6, baseline 196 → 192, ISP 누적 net 0)
- [x] **ARCHITECTURE.md 갱신 항목 명시** (§13.5 V-?? cold_ext_namespace entry + §13.8-G register sub-list REVOKED + §13.8 정책 수 5 유지)
- [x] **escalate 3건** (Q1 closure API / Q2 secondary.rs 모듈 처분 / Q3 partition_workspace 무관 확인)
- [x] **implementer prompt 초안** 본문 포함 (Phase 2~5)
- [x] **본 라운드 자체는 코드 무수정** — arch/ 문서만 변경
- [x] **§J 본문 무수정 / spec INV-LAYER-001 비고 무수정** 명시 (B-5b 결정 보존)
- [x] **ISP 누적 net 0 변동 분석** (Phase 2 +1 / Phase 5 -1)
- [x] **사용자 의도 반영**: 절충안 (hot path 보존 + cold path string lookup) + researcher 조사 (ggml `get_proc_address` 직역) + R2 게이트 (KIVI hot path 제외)
