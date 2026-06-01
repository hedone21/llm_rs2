# Handoff — 설계 구체화 트랙 종료 (전 설계 게이트 닫힘 → Phase α 코드)

**작성**: 2026-06-02 (G5 grill + G4 게이트 해제 — 설계 트랙 종료)
**HEAD(설계 content)**: `0e5cb946 docs(arch): G5 detail-fill — DeviceTarget {Cpu,Gpu,Npu} + StepInfo 3필드 확정` (본 handoff commit 이 위에 1개)
**브랜치**: `master`
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"Phase α-W 진행"** (`hardware.rs`/`pipeline.rs`/`capability/` 신설 + `stages/` 골격 — G5 확정 타입 동봉). **설계 게이트 전부 닫힘 — 단일 코드 경로.**

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. v1(동결) = `arch/pipeline_stage_design.md`. **코드 아님 — 설계 문서 구체화 트랙.** Phase α-W/α-K = 코드 진입(아직 아님).

---

## TL;DR

G5 grill (Q1~Q2) 완료 → arch doc §2.1 L178/179 + §3.5/§5.1 코드블록 타입 정의 + 連歴 L200 반영·커밋(`0e5cb946`). `DeviceTarget`=`{Cpu,Gpu,Npu}` 추상 역할 + `StepInfo`=`{pos,decode_step,pressure}` 3필드 Copy 확정. 이로써 **모든 설계 게이트(G1/Hardware/G3/G3-reconcile/G5/ResilienceStrategy) 닫힘**. 마지막 미결이던 **G4(수치 임계)는 게이트 해제** — `band()` cutoff 은 설계 결정이 아니라 Phase α-W 코드 시점 튜닝 상수(default = 기존 `MemoryStrategy` 이산값 carry)로 강등, 정의 불명이던 R3/R4 의존 제거. **설계 구체화 트랙 종료** — 남은 건 코드(Phase α-W → α-K)뿐. 멈춘 이유 = 트랙 종료, 다음 세션은 코드 진입.

---

## pressure/ 해체 — 닫힌 결정 (doc 반영·커밋 완료, **재grill 금지**)

| # | 결정 |
|---|---|
| Q1 | `pressure/` → **`kv/`** (KV-cache 도메인: format **flat** + `eviction/` policy + `offload/` tier + `d2o/` algo) |
| Q2 | `pressure/weights/` → **`weight/`** 신설 (runtime swap 오케스트레이션. §13.8-O `RuntimeResourcesAccess` trait 경계 유지, `models/weights/` load-time artifact 와 분리) |
| Q3 | handler split = **함수 단위 cut** (트리거 `handle()`→`stages/`, 알고리즘→도메인 dir; d2o merge ~440 LOC·`offload_one`/`recall_one` 추출). G3 "file 단위" 정밀화 |
| Q4 | format = `kv/` **1차 타입(flat)**, `kv/format/` subdir 철회. format=**축**(L2 trait `KVCacheFormat`/`WeightFormat` + 공유수학 `quant/`), kv 종속 아님. per-layer 동적 precision = Stage+format mutation primitive |

**근거 핵심 (재확인 불요)**: redesign 후 pressure/ 내용물 전부 KV-cache 데이터 → "pressure" 는 역사적 사고. `weights/` 는 KVCache 0 import 오배치 입주자. format impl 은 데이터 내재(KIVI KV cache ↔ Q4 weight block 공유 코드 0, `quant/` 만 공유). cache_manager(1529) = 전부 orchestration → `session/` PipelineRegistry(L4), kv/ 잔여 0.

---

## G5 detail-fill — 닫힌 결정 (doc 반영·커밋 완료, **재grill 금지**)

| # | 결정 |
|---|---|
| Q1 | `DeviceTarget` = `enum { Cpu, Gpu, Npu }` 추상 연산 역할. 구체 backend(OpenCL/CUDA feature 배타)는 registry resolve, `--opencl-rpcmem` 은 device 아님(memory interop), `Npu` 는 backend 부재(qnn 제거)지만 partition `SliceSpec` 가 spec 전제. `{Cpu,OpenCL,OpenCLRpcmem,Cuda}` 구체 enum 은 registry 중복+dead variant 라 기각, `Gpu(u8)` 다중GPU 는 YAGNI |
| Q2 | `StepInfo` = `{ pos, decode_step, pressure }` 3필드 `Copy`(borrow 0). v1 `StepCtx`(`session/traits.rs:21`) 5필드 중 `pos`/`decode_step` 승계. 드롭: `prev_token`(샘플러 도메인), `kv_capacity`(held KVCacheFormat handle query — god-ctx 회피), `stop_requested`(→`StageOutcome::Stop` 반환). `prev_token` 은 observe Stage 승격 trigger(driver 이미 보유 → 확장 ripple 0) |

doc 위치: §2.1 L178/179(표) · §3.5 L354-361(`DeviceTarget` enum+근거) · §5.1 L526-535(`StepInfo` struct+근거) · 連歴 L200.

---

## G4 (수치 임계) — 게이트 해제 (트랙 종료 결정 2026-06-02, **재논의 불요**)

`Pressure.band()` cutoff 은 **설계 게이트가 아니라 Phase α-W 코드 시점 튜닝 상수**다. 근거:
1. **설계는 이미 확정** — 4-level `PressureLevel`→`Pressure(0–100)`+`band()` 흡수, `MemoryStrategy` 소멸 (doc §5.1 L541). 남은 건 cutoff **숫자**뿐인데 상수를 design doc 에 박는 건 안티패턴(튜닝 불가화).
2. **day-1 default 존재** — 기존 이산 임계(`Warning 0.85 / Critical 0.50 / Emergency 0.25` target ratio)를 0–100 구간 매핑으로 carry 하면 동작 보장.
3. **실측 튜닝은 코드 시점** — cutoff 근거(디바이스 메모리 압력 실측)는 Phase α 에서 실제 데이터로 조정.
4. **phantom 의존 제거** — 게이트가 묶여 있던 **R3/R4 가 어떤 doc/handoff 에도 미정의**(계보를 거치며 참조만 전파, 정의 부재). 정의 없는 게이트는 다음 세션을 또 "R3/R4가 뭐지?"로 끌고 가므로 끊음.

→ **band() default = 기존 MemoryStrategy 이산 임계 carry. 코드 상수이지 설계 미결 아님.**

---

## 다음 작업 (순서)

1. **Phase α-W** (진입 문장, 코드 첫 진입) — `hardware.rs`(`Hardware`+`DeviceTarget` G5 확정) / `pipeline.rs`(`PipelineStage`+`StepInfo` G5 확정+`Pressure`+`PressureSource`) / `capability/` 신설 + `stages/` 3-way 골격 + `CommandSource`/`CommandDispatcher`/`LoopControl` + `ResilienceAction`/`MemoryStrategy` 삭제. **`band()` default = 기존 MemoryStrategy 이산 임계 carry**(G4 해제). 검증 게이트: 호스트 `cargo build`+`cargo test` 통과, S25 OpenCL TBT 회귀 0(bit-identical).
2. **Phase α-K** (코드, α-W 후) — `pressure/`→`kv/` + `pressure/weights/`→`weight/` rename **189 ref/53 file** + handler 함수 cut(d2o merge ~440 LOC·`offload_one`/`recall_one`) + `KVCacheFormat`/`WeightFormat` trait 확립.

---

## Landmines / 미해결

- **kv//weight/ rename = Phase α-K 코드 작업, 아직 미적용** — doc만 반영됨. 코드 디렉토리는 여전히 `engine/src/pressure/`(+ `pressure/weights/`). rename blast radius **189 ref/53 file**(기계적). d2o_handler 2273 LOC 함수 cut(트리거/알고리즘/테스트 분리) 동행.
- **`kv_migrate.rs`** = §4.1 연혁 storage-slot 트랙(🟡) 미해소. kv/ 로 가지만 최종 거처는 KV storage-slot 분리(weight `Arc<LayerSlot>` 대칭) 후 재검토.
- **G4 = 게이트 아님(해제됨)** — `band()` cutoff 은 Phase α-W 코드 상수(default = MemoryStrategy 임계 carry). 과거 "R3/R4 게이트" 표현은 미정의 참조라 폐기. 다음 세션은 R3/R4 추적하지 말 것.
- **커밋 금지 대상**(working tree untracked): `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_full_matrix_2026_05_28/*`, `.agent/todos/handoff_microbench_full_matrix_p4_sweep_wait_2026_05_29.md`. 커밋 시 **파일명 지정**(`git add -A` 금지).
- **review 스킬 트리거 = 한국어 '리뷰' 만**. "확인/논의/검토/비교"는 일반 grill/분석 모드.

## 선행 문서
① 본 handoff ② SSOT `arch/pipeline_stage_design_v2.md`(§0.4 front-door·§2.1 type→file + G3-reconcile 連歴·§5.4 resilience 2-source·§3.5 Hardware·§3.6 wiring) ③ v1 동결 `arch/pipeline_stage_design.md` ④ 용어 `/CONTEXT.md`(3축: stage⊥format⊥hardware) ⑤ 코드 근거: `engine/src/pressure/`(현 위치, Phase α-K rename 대상), `engine/src/quant/`(공유 format 수학), `shared/src/lib.rs:189`(`EngineCommand`), `engine/src/pressure/cache_manager.rs`(→ L4 PipelineRegistry).

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "Phase α-W 진행"
- 왜 멈췄나: ✓ G4 게이트 해제 → 설계 구체화 트랙 종료, 다음은 코드(Phase α-W) 단일 경로
- 최대 landmine: ✓ kv//weight/ rename 은 doc만 — 코드(Phase α-K) 미적용. Phase α-W 는 신설 위주(α-K rename 과 분리)
- 검증: Phase α-W 부터 **코드 게이트** — `cargo build`+`cargo test`+S25 TBT 회귀 0
