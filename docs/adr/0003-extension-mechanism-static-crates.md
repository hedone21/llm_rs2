# ADR-0003: 확장 메커니즘 — 정적 링크 technique crate + 자동 등록, 런타임 `.so` 보류

> **Status**: Accepted
> **Date**: 2026-06-05
> **Decision-makers**: 사용자 + Architect (grill-with-docs 세션 "plugin 확장 목표 정렬", Q1~Q6)
> **Selected**: 기능 확장(h2o/d2o/turboquant 등)은 **정적 링크된 technique crate**가 rich Rust trait을 구현하는 방식. **per-axis 확장 표면**(stage/format/hardware 독립). 등록은 **crate 패키징 + workspace glob + 자동 등록(linkme distributed_slice + startup self-test)**. 런타임 `.so`(C-ABI)는 *eventual north star*로 보류.
> **Related**: `arch/pipeline_stage_design_v2.md` §합격선(path-dependent OCP)·`PipelineRegistry`·`CapabilityRegistry`, `/CONTEXT.md` 3축 직교(M+N+K), ADR-0001(KV dispatch paradigm), ADR-0002(Pressure)

---

## 1. Context

최종 목표: "**zero-compile로 plugin 설치를 통한 기능 확장**" — h2o·tensor partition·d2o·turboquant 같은 기법을 (1) 명확한 API, (2) 명확한 hooking 포인트, (3) OCP로 추가. 사용자는 현 설계가 이 목표에서 부족하다고 느꼈다.

grill 과정에서 "zero-compile"의 실현 수단을 코드 사실에 대고 검증했다:

- **세 hot-path 축의 trait이 C-ABI를 못 넘는다** (workflow surface-map, 2026-06-05): `EvictionPolicy`(eviction.rs:13)는 `&mut KVCache`(내부 `Arc<dyn Buffer>`+`Arc<dyn Backend>`)를, `KVCacheFormat`(kv_cache_format.rs:57)은 `&Tensor`/`&dyn Backend`를, `Backend`(backend.rs:128, 60+ 메서드)는 `Arc<dyn Backend>`/`ocl::Event`를 싣는다. 전부 trait object·fat pointer·Rust enum이라 C-ABI 직통 불가.
- **format 축은 paired kernel을 `.so`로 못 싣는다**: 커널은 `include_str!`로 엔진 init에 GPU 드라이버 대상 컴파일된다 — `.so` format 플러그인이 자기 `.cl`을 실어 보낼 경로가 없다.
- **이 repo 프로필이 `.so`에 불리**: `lto = "fat"`, `codegen-units = 1`(`/Cargo.toml`)은 hot-path 인라인을 위해 존재하는데, 동적 경계는 cross-crate 인라인을 막는다(per-token format/hardware 축에 특히 손해). `panic = "abort"`는 플러그인 격리를 어렵게 한다. 타겟이 ARM64 Android인데 프로덕션 dlopen은 SELinux/W^X로 제한된다.
- **기존 `.so` 선례(HTP FastRPC, htp_fastrpc.rs)는 정반대 방향**: vendor `.so`는 Rust trait을 구현하지 않고, Rust가 `Backend`를 구현한 뒤 그 안에서 `extern "C"`에 raw pointer/C-struct만 넘긴다. 풍부한 타입은 엔진에 남는다.
- **OCP를 막는 실체는 closed match arm**: 새 eviction 정책 추가가 `session/chat/session.rs:621` `match args.eviction_policy`를, 새 backend가 `session/init.rs:203` `match args.backend`를 편집하게 강제한다. 한편 arch v2는 이미 "cold path = zero-edit OCP(registration 1줄 제외)"를 설계했고 `CapabilityRegistry`(capability.rs:39)·`BackendRegistry`(hardware.rs:36)는 배선됐으나, stage용 `PipelineRegistry`는 **설계만 있고 미배선**이다.

## 2. Decision

**(D1) 정적 우선, `.so` 보류.** 기능 확장은 *정적 링크된 Rust crate*가 rich trait(`EvictionPolicy` 등)을 구현하는 방식으로 한다. C-ABI 평탄화 비용(rich 타입 손실 + format 커널 배송 + Android dlopen + LTO 인라인 손실)이 "엔진 재빌드 없는 A/B 교체" 이득을 현시점 초과한다. 런타임 `.so`는 폐기가 아니라 **보류된 north star**다 — (D3)의 crate 패키징이 `cdylib` 승격으로 가는 활주로를 미리 깐다.

**(D2) per-axis 확장 표면.** stage/format/hardware는 직교(`/CONTEXT.md` M+N+K)이므로 단일 통합 plugin ABI를 만들지 않는다. 축마다 다른 표면:
- **stage = "planning" 표면** — 기법은 캐시를 읽고 **계획**을 반환하고 **엔진이 실행**(compaction + 가중 합산). 버퍼를 직접 안 만진다. h2o·sliding·streaming·h2o+·**d2o**(merge 포함)를 한 표면으로 덮는다. **(정정 — ADR-0004)**: 이 표면의 정식 형태는 단일 trait **`KVCacheStage` → `KVCachePlan`**(keep `KeepSpec{LayerWide|PerHead}` + `WeightedMerge`)이다. 본 §D2 작성 시 "planning 표면이 h2o+/d2o 를 덮는다"는 *낙관적*이었고(초기 `plan_keep` 는 layer-wide keep 전용이라 h2o+=None·d2o=별도 handler), ADR-0004 가 per-head keep + 가중 merge + stateful impl + `StageCtx` 읽기 추상으로 *어떻게* 덮는지 확정한다.
- **format = accessor + paired-kernel 표면** — 데이터 레이아웃 소유 + 전용 커널 필요. turboquant·KIVI가 여기. 가장 무겁고 별도·후순위.
- **hardware = accessor 표면** — backend(HTP를 뒤집은 형태).

**(D3) 자동 등록 = technique crate + workspace glob + linkme.** 각 기법 = `crates/techniques/*` 아래 별도 crate. workspace member glob이 자동 발견. 엔진 bin이 각 crate를 의존(→ 링크 보장). crate는 공유 `technique-api`의 `#[distributed_slice]`에 자기를 등록(linkme). 엔진은 construction 시 그 슬라이스를 읽어 정책을 고른다. **startup self-test**가 기대 기법 수/이름을 단언해 fat-LTO `--gc-sections`의 silent section-strip을 fail-fast로 잡는다.

**(D4) "코드 수정 0"의 정의.** 기법 추가 = 기존 *로직* 0 edit + **Cargo 매니페스트 의존성 1줄**. 매니페스트 1줄은 *코드*가 아니라 *설정*이므로 OCP를 위반하지 않는다.

> **(정정 — M3 실측 2026-06-05)**: 의존성 1줄만으로는 부족하다. Rust 는 **미참조 의존 rlib 을 링크에서 제외**하므로(dead-crate elision), technique crate 의 `#[distributed_slice]` 등록이 바이너리에 포함되지 않는다(실측: dev-dep 선언 후 `find_stage` 가 `None`). 활성화하려면 **force-link 참조 1줄**(`use <technique_crate> as _;`)을 designated 지점(예: 엔진의 technique-link 모듈 / bin)에 추가해야 한다. 따라서 실제 확장 비용 = **기존 로직 0 edit + dep 1줄 + force-link 1줄**(둘 다 기계적 설정성 라인이라 OCP 유지). 아래 §3 의 "crate Cargo 의존이 링크를 보장"은 낙관적이었음 — 의존은 *컴파일 가용성*만 보장하고 *링크 포함*은 force-link 참조가 보장한다.

## 3. Rationale

- **rich 타입 보존**: 정적 링크면 `&mut KVCache`·`Arc<dyn Backend>`를 그대로 경계로 쓴다 — C-ABI 평탄화(`Merge` CSR화, opaque handle, accessor table)를 *지금* 안 내도 된다.
- **per-axis가 직교성을 보존**: 통합 ABI는 가장 무거운 표면(format accessor+kernel)으로 수렴해 stage의 순수-함수 깔끔함을 잃는다. 축별 표면은 `/CONTEXT.md`의 M+N+K 가산성과 일치.
- **linkme > inventory > 명시 카탈로그**: inventory는 life-before-main 생성자라 `panic="abort"`/Android에 위험. 명시 중앙 카탈로그(model 1)는 "폴더만 떨구기"를 깬다. linkme는 순수 링커 데이터(생성자 0)라 안전하고, crate Cargo 의존이 링크를 보장해 linkme의 dead-crate 함정을 제거한다. 남은 fat-LTO section-strip 리스크는 self-test로 관리. *(정정 — M3 실측: Cargo 의존만으로는 dead-crate 함정이 **제거되지 않는다**. force-link 참조 1줄이 추가로 필요 — D4 의 정정 노트 참조.)*
- **crate 격리 + `.so` 활주로**: 외부 기여자가 engine 내부를 못 망가뜨리는 경계이자, 나중에 `cdylib`로 빌드해 진짜 `.so`로 가는 직행로 (D1의 north star와 정합).
- **설계는 이미 맞다**: arch v2가 zero-edit-OCP·`PipelineRegistry`·mechanism-over-policy(커널 모듈/커스텀 allocator 계약)를 설계했다. 본 ADR은 그 설계의 *수단*(정적 crate + linkme)을 확정하고 미배선 부분을 작업으로 지목한다.

## 4. Consequences

**구현 갭 (작업 대상):**
- stage `PipelineRegistry` 신설 → `session.rs:621` match arm 제거. backend selection도 동일(`init.rs:203`).
- format wiring seam(α-K, forward_gen_fmt.rs) 배선. d2o를 `plan_keep`(현재 미사용; D2OHandler가 직접 mutate)로 이전 + `Merge`(`{into, from: Vec<usize>}`)에 가중치 필드 추가.
- 기법들을 `crates/techniques/*` crate로 재패키징 + `technique-api` 공유 crate 신설 + workspace glob.
- **기여자 문서**(3번째 요구): "기법 추가법" — hook 지점·trait 시그니처·등록을 외부 기여자가 한눈에 보는 형식. 생성/등록된 카탈로그가 곧 기법 인덱스.

**수용한 trade-off:**
- 기법 추가/교체에 엔진 재빌드 필요(정적 링크의 본질). "zero-compile A/B 교체"는 `.so` 승격(D1 north star)까지 보류.
- format 축(turboquant/KIVI)은 stage처럼 깔끔하지 않다 — accessor + 커널 배송 문제는 별도 후속 결정.

**검증:** startup self-test(기대 기법 등록 단언)가 fat-LTO/gc-sections silent drop의 게이트. self-test가 LTO 문제를 드러내면 linkme→build.rs codegen(명시 참조 배열, GC 무관)으로 폴백.

## 5. Alternatives Considered

### (a) 정적 crate + linkme 자동 등록 (**ACCEPTED**)
§2. 채택 사유 §3.

### (b) 런타임 `.so` (C-ABI) — 즉시 도입 (REJECTED, 보류)
진짜 zero-compile·언어 자유·디바이스 배포. **거부(현시점)**: rich 타입이 C-ABI를 못 넘어 축마다 평탄 capability 표면 신설 필요 + format은 커널 배송 미해결 + fat-LTO 인라인 손실 + Android dlopen 제약. 이득(재빌드 없는 교체) < 비용. **north star로 보류** — crate 패키징이 `cdylib` 승격 활주로.

### (c) `dylib` (Rust ABI) lockstep (REJECTED)
풍부한 타입 유지하며 동적 로딩. **거부**: Rust는 stable ABI 부재 → rustc/dep/profile 불일치 시 silent UB(이 repo는 `rust-toolchain.toml`도 없어 미고정). fat-LTO 불가, `panic="abort"` 격리 불가. 호스트 벤치 A/B 외엔 부적합.

### (d) 단일 통합 plugin ABI (REJECTED)
세 축을 한 인터페이스로. **거부**: 가장 무거운 표면으로 수렴해 stage 순수성 상실. 직교(M+N+K) 위배.

### (e) 명시 중앙 카탈로그 등록 1줄 (REJECTED) / (f) inventory (REJECTED) / (g) 모듈+build.rs glob (REJECTED)
(e) "폴더만 떨구기" 위배. (f) life-before-main이 `panic="abort"`/Android에 위험. (g) literally-0-edit지만 crate 격리·`.so` 활주로 없음. → (D3) crate+linkme가 격리·활주로·near-zero(매니페스트 1줄)를 동시 충족.

## 6. References

- `arch/pipeline_stage_design_v2.md` §합격선(line 24·129 path-dependent OCP), `PipelineRegistry`/`CapabilityRegistry`/backend factory, mechanism-over-policy(§5.3)
- `/CONTEXT.md` — 3축(stage⊥format⊥hardware) 직교 M+N+K
- 확장 trait 표면: `engine/src/kv/eviction.rs:13`(`EvictionPolicy`), `engine/src/format/kv_cache_format.rs:57`(`KVCacheFormat`+`Merge`:46), `engine/src/backend.rs:128`(`Backend`)
- closed match arm(OCP 갭): `engine/src/session/chat/session.rs:621`, `engine/src/session/init.rs:203`
- 배선된 레지스트리: `engine/src/capability.rs:39`(`CapabilityRegistry`), `engine/src/hardware.rs:36`(`BackendRegistry`)
- `.so` 선례: `engine/src/backend/htp_fastrpc.rs`(libloading FFI), `engine/src/backend.rs:1319`(`get_extension`)
- `/Cargo.toml` — `lto="fat"`/`codegen-units=1`/`panic="abort"`(LTO·격리 제약 근거)
- ADR-0001(KV dispatch paradigm, 선행), ADR-0002(Pressure)
