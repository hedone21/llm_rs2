# ADR-0002: Pressure 스칼라 — lossy 단일화 (graded 입력 융합 ⊥ mode 출력 분리)

> **Status**: Accepted
> **Date**: 2026-06-02
> **Decision-makers**: 사용자 + Architect (R1 grill, "C1·C2·R1" 세션)
> **Selected**: 갈래 (a) — 전 센서 magnitude 를 단일 `Pressure(0–100)` 스칼라로 융합(graded 입력), `switch`/`suspend` 같은 *mode* 출력은 이산 `EngineCommand` 채널 유지
> **Related**: `arch/pipeline_stage_design_v2.md` §5.1 / §5.4 (R1 연혁), `/CONTEXT.md` "Pressure / PressureSource", ADR-0001 후속, `spec/41-invariants.md` §3.28
> **Supersedes (부분)**: v2 §5.4 R-5 연혁 점 (3) *"Thermal/Energy/Compute 만 이산 잔존, Memory 는 graded 라 소멸"* (2026-06-01)

---

## 1. Context

엔진의 resilience 입력은 `Pressure(0–100)` 단일 scalar 로 표현된다 (`StepInfo` 에 per-step read-only 로 실려 Stage 가 `band()` 로 반응). 그런데 설계 문서가 *무엇이 이 스칼라를 채우는가*에 대해 스스로 모순됐다:

- **§5.1 + `/CONTEXT.md`**: "memory/thermal/energy 융합 → 단일 scalar" (3신호 합산).
- **§5.4**: "Memory 만 graded scalar, Thermal/Energy/Compute 는 이산 채널 전용" (thermal/energy 를 scalar 에서 배제).

코드가 §5.1 을 반증했다 — 같은 `Level::Warning` 에서 전략별 산출 액션이 다르다:

```
engine/src/resilience/strategy/memory.rs:39   Warning → Evict { target_ratio: 0.85 }   (graded knob)
engine/src/resilience/strategy/thermal.rs:34  Warning → SwitchBackend { ... }          (discrete mode)
engine/src/resilience/strategy/energy.rs:29   Warning → SwitchBackend { ... }          (discrete mode)
```

세 신호가 같은 강도에서 **서로 다른 동작**을 요구하므로, 하나의 scalar 로 융합하면 소비 Stage 가 `pressure=90` 을 읽고도 evict 할지 switch 할지 알 수 없다 (정보 손실). 이 모순은 Phase α-W 가 `PressureSource`/`LocalPressureSource`/`CommandSource` 를 신설하기 전에 닫혀야 했다 — "스칼라에 무엇이 들어가나" 가 미정인 채로는 `LocalPressureSource` 를 정확히 구현할 수 없다.

## 2. Decision

**모순의 근원은 "스칼라 *입력*" 과 "*mode* 출력" 을 혼동한 것이다. 둘을 분리해 해소한다:**

- **입력 융합** — `LocalPressureSource` 가 memory·thermal·energy 등 모든 system 압력의 *magnitude* 를 하나의 `Pressure(0–100)` 으로 융합한다. 이 스칼라가 graded 응답(eviction 강도)을 구동한다.
- **출력 분리** — `switch`/`suspend` 처럼 scalar 로 환원 불가한 *mode* 동작("73만큼 switch" 는 없다 — on/off)은 이 스칼라가 아니라 별도 이산 `EngineCommand` 채널(`CommandSource` → `CommandDispatcher`)로 흐른다.
- **결과**: thermal 은 *입력 scalar*(magnitude 융합 → graded eviction 구동) ⊕ *출력 이산명령*(switch) 양쪽에 관여한다. Memory 는 mode 출력이 없어 graded(연속 scalar) 전용. `LocalPressureSource`(전 센서 → scalar)와 `LocalPolicy`(센서 → mode 명령)는 같은 thermal/energy 센서를 두 목적으로 읽는다.

**이 융합은 의도적으로 lossy 하다** — 소비 Stage 는 압력의 *출처*(memory? thermal?)를 구분하지 않는다. 단순함과 관리 용이를 위해 출처 정보 손실을 명시적으로 수용한 trade-off다.

## 3. Rationale

- **모순 해소**: §5.1(입력 융합)과 §5.4(이산 mode 출력)는 사실 *다른 차원*을 말하고 있었다 — 입력 aggregation ⊥ 출력 routing. 분리하면 둘 다 참이 된다.
- **"no anymap" 정당화가 정직해짐**: 신호 *종류* 확장(typed anymap)이 불필요한 이유가, 이전엔 "3신호를 융합하니까"(hollow — 실제로는 안 융합됐었음)였으나, 이제 "출처 손실을 의도적으로 수용하니까"(명시적 trade-off)로 전환됐다.
- **이산 채널은 어차피 필요**: `CommandSource`/`EngineCommand` 는 manager IPC(manager 가 fine-grained 명령 송신)용으로 존재해야 한다. mode 출력을 여기로 보내는 건 새 복잡도가 아니라 있던 채널의 역할 명확화다.
- **physical 제약**: switch(GPU→CPU)는 mode 라 스칼라 숫자로 표현 불가. 따라서 "완전 단일 스칼라" 는 물리적으로 불가능하다.

## 4. Consequences

**수용한 lossy (의도적):**
- eviction 이 thermal/energy 압력에도 반응한다 → 토큰을 버려도 칩은 안 식는다 (무용하나 무해).
- 소비 Stage 는 `pressure=90` 의 출처를 구분 못 한다.

**구현 함의 (Phase α-W):**
- 현 `LocalPressureSource`(= 구 `cache_manager.rs::determine_pressure_level`, memory 만 계산)를 **전 센서 융합**(memory+thermal+energy → 0–100)으로 확장해야 한다. 이는 R-5 연혁의 "manager-less 이산 정책은 로컬 센서 모니터 인프라를 동반" 미구축 의존성과 연결된다.

**Promotion-trigger (재고 조건):** 출처 분별이 필요한 정책(예: thermal 만 반응하는 graded stage)이 등장하면, 그때 `Pressure` 에 source-tag 를 도입하거나 신호별 채널로 분화한다. 그 전엔 단일 lossy scalar 를 유지한다 (YAGNI).

## 5. Alternatives Considered

### 갈래 (a) — 입력 융합 + mode 출력 분리 (**ACCEPTED**)
§2 의 결정. 채택 사유 §3.

### 갈래 (b) — 완전 단일 채널 (REJECTED)
이산 채널을 manager IPC 로만 한정하고, manager-less autonomous 의 switch/suspend 를 `Pressure` threshold-trigger Stage 로 발동. **거부 사유**: 스칼라가 출처를 못 담으므로, threshold Stage 가 *메모리* 압력에 switch 를 오발동할 수 있다 (switch 는 메모리엔 무용·유해 — GPU→CPU 전환은 메모리를 안 줄이고 속도만 떨어뜨림). 이는 §5.4 연혁의 **"manager-less 이산 정책(switch/suspend)이 1급 요구(사용자 확정)"** 의 *정확성*을 깨뜨린다.

### 갈래 (c) — 출처 분별 (source-tag / typed anymap) (REJECTED, 현 시점)
`Pressure` 에 출처 태그를 달거나 신호별 typed 채널 유지. **거부 사유**: 현재 graded 응답 소비자가 eviction 하나뿐이라 출처 분별의 실수요가 없다 (deletion test). 단순함을 위해 lossy 수용. 실수요 등장 시 promotion-trigger 로 재도입.

## 6. References

- `arch/pipeline_stage_design_v2.md` §5.1 (pressure carrier), §5.4 (2-source 모델 + R1 연혁)
- `/CONTEXT.md` — "Pressure / PressureSource" 항목 (sharpened 2026-06-02)
- `engine/src/resilience/strategy/{memory,thermal,energy,compute}.rs` — 전략별 산출 액션 (모순 근거)
- ADR-0001 — KV dispatch paradigm (선행)
