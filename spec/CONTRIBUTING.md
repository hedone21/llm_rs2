# Spec 관리 체계 가이드

> **대상 독자**: llm_rs2 아키텍트, 구현자, 테스터.
> 이 문서를 읽으면 spec 변경, arch 문서 작성, 테스트 추가를 즉시 시작할 수 있다.

---

## 1. 3계층 구조 개요

llm_rs2의 설계 지식은 3개 디렉토리에 계층적으로 관리된다.

```
spec/       ← 불변. WHAT (무엇을 보장해야 하는가)
arch/       ← 가변. HOW (코드에서 어떻게 구현하는가)
tests/spec/ ← 검증. DOES IT HOLD? (실제로 보장되는가)
```

| 계층 | 성격 | 내용 | 변경 빈도 |
|------|------|------|----------|
| **spec/** | 불변, 구현 독립적 | 요구사항, 불변식, 인터페이스, FSM 전이 테이블, 의사코드 | 낮음 (설계 변경 시만) |
| **arch/** | 가변, 코드 종속적 | 코드 파일 경로, config 키, CLI 플래그, Default 값, struct 매핑 | 높음 (코드 변경마다) |
| **tests/spec/** | 검증 | spec/의 INV/요구사항을 자동 테스트로 검증 | 중간 (INV 추가 시) |

**핵심 원칙**: spec/은 "무엇"만, arch/는 "어떻게"만. spec/에 코드 경로가, arch/에 불변식이 등장하면 잘못이다.

### 1.1 각 계층의 예시

**spec/22-manager-algorithms.md** (WHAT):
```
[INV-031] integral은 항상 [0, integral_clamp] 범위 내이다. (MUST)
```

**arch/22-manager-algorithms.md** (HOW):
```
INV-031 구현 위치: manager/src/policy/pi_controller.rs:87
구현 방법: f64::clamp(self.integral, 0.0, self.config.integral_clamp)
Config 키: policy.pi.integral_clamp (기본값: 2.0)
```

**tests/spec/test_inv_031.rs** (DOES IT HOLD?):
```rust
#[test]
fn test_inv_031_integral_always_in_range() {
    // integral_clamp = 2.0인 PI Controller에
    // measurement = 1.0을 1000회 반복 입력해도
    // integral이 [0, 2.0]을 벗어나지 않는지 검증
}
```

---

## 2. 단일 ID 관통 원칙

모든 요구사항과 불변식에는 고유 ID가 부여된다. 이 ID는 3계층을 관통하여 추적성을 보장한다.

### 2.1 관통 흐름

```
spec/22-manager-algorithms.md  →  arch/22-manager-algorithms.md  →  tests/spec/test_inv_030.rs
       [INV-030] 정의                [INV-030] 구현 기술               test_inv_030_*() 검증
```

- `INV-030`이 spec/에서 정의되면, arch/에서 구현 위치를 기술하고, tests/spec/에서 `test_inv_030`으로 검증한다.
- ID 검색 한 번으로 정의-구현-검증 전체를 추적할 수 있어야 한다.

### 2.2 ID 접두사 체계

모든 요구사항 ID는 `[PREFIX-NNN]` 형식이다. 접두사별 할당 파일과 범위는 다음과 같다.

| 접두사 | 할당 파일 | 번호 범위 | 실제 사용 범위 | 설명 |
|--------|----------|----------|--------------|------|
| `SYS` | 00-overview.md | 001~065 | SYS-001 ~ SYS-065 | 시스템 목표, 용어, 모델 지원 |
| `SYS` | 01-architecture.md | 070~099 | SYS-070 ~ SYS-099 | 크레이트 분해, 서브시스템, 배포 |
| `PROTO` | 10-protocol.md | 010~075 | PROTO-010 ~ PROTO-075 | Wire format, transport, lifecycle |
| `MSG` | 11-protocol-messages.md | 010~100 | MSG-010 ~ MSG-100 | 메시지별 필드 정의 |
| `SEQ` | 12-protocol-sequences.md | 010~104 | SEQ-010 ~ SEQ-104 | 정규 상호작용 시퀀스 |
| `MGR` | 20-manager.md | 010~048 | MGR-010 ~ MGR-048 | Manager 개요, HLR |
| `MGR` | 21-manager-state.md | 050~087 | MGR-050 ~ MGR-087 | Manager 상태 머신 |
| `MGR-ALG` | 22-manager-algorithms.md | 010~072 | MGR-ALG-010 ~ MGR-ALG-072 | PI, Supervisory, ActionSelector, Relief |
| `MGR-DAT` | 23-manager-data.md | 010~060 | MGR-DAT-010 ~ MGR-DAT-060 | 설정 스키마, 센서 인터페이스 |
| `ENG` | 30-engine.md | 010~073 | ENG-010 ~ ENG-073 | Engine 개요, HLR |
| `ENG-ST` | 31-engine-state.md | 010~070 | ENG-ST-010 ~ ENG-ST-070 | Engine 상태 머신 |
| `ENG-ALG` | 32-engine-algorithms.md | 010~095 | ENG-ALG-010 ~ ENG-ALG-095 | KV 캐시, eviction, 양자화 |
| `ENG-DAT` | 33-engine-data.md | 010~080 | ENG-DAT-010 ~ ENG-DAT-080 | 텐서 레이아웃, 캐시 포맷 |
| `CROSS` | 40-cross-cutting.md | 010~092 | CROSS-010 ~ CROSS-092 | 에러 처리, 로깅, 타이밍 |
| `INV` | 41-invariants.md (수집) | 001~085 | INV-001 ~ INV-085 | 전체 불변식 카탈로그 (65개) |
| `CON` | 각 파일 내 Constraints | 010~042 | CON-010 ~ CON-042 | 적합성 기준 |

### 2.3 INV-ID 할당 범위

INV는 정의 위치(원본 파일)에 따라 번호 대역이 나뉜다.

| INV 범위 | 원본 파일 | 개수 | 분류 |
|----------|----------|------|------|
| INV-001 ~ INV-006 | 00-overview.md | 6 | System |
| INV-010 ~ INV-018 | 01-architecture.md | 9 | Architecture |
| INV-020 ~ INV-028 | 10-protocol.md, 11-protocol-messages.md | 9 | Protocol |
| INV-030 ~ INV-051 | 22-manager-algorithms.md | 22 | Manager Algorithm |
| INV-060 ~ INV-065 | 30-engine.md | 6 | Engine Architecture |
| INV-070 ~ INV-076 | 31-engine-state.md | 7 | Engine State Machine |
| INV-080 ~ INV-085 | 41-invariants.md (신규) | 6 | Cross-cutting |
| **합계** | | **65** | |

**다음 가용 번호**: INV-086 (cross-cutting), INV-019 (architecture), INV-029 (protocol), INV-052 (manager algorithm), INV-066 (engine), INV-077 (engine state)

### 2.4 Constraint (CON-*) 할당 범위

| CON 범위 | 원본 파일 | 내용 |
|----------|----------|------|
| CON-010 ~ CON-012 | 10-protocol.md | 프로토콜 적합성 |
| CON-020 ~ CON-022 | 11-protocol-messages.md | 메시지 호환성 |
| CON-030 ~ CON-033 | 12-protocol-sequences.md | 시퀀스 적합성 |
| CON-040 ~ CON-042 | 40-cross-cutting.md | 횡단관심사 적합성 |

### 2.5 Manager/Engine Constraint (MGR-C*, ENG-ALG-C*) 할당 범위

| 접두사 | 범위 | 원본 파일 |
|--------|------|----------|
| MGR-C01 ~ MGR-C02 | 20-manager.md | Manager 개요 제약 |
| MGR-C03 ~ MGR-C05 | 21-manager-state.md | Manager 상태 제약 |
| MGR-C06 ~ MGR-C10 | 22-manager-algorithms.md | Manager 알고리즘 제약 |
| MGR-DAT-C01 ~ MGR-DAT-C02 | 23-manager-data.md | Manager 데이터 제약 |
| ENG-ALG-C01 ~ ENG-ALG-C08 | 32-engine-algorithms.md | Engine 알고리즘 제약 |
| ENG-DAT-C01 ~ ENG-DAT-C07 | 33-engine-data.md | Engine 데이터 제약 |

### 2.6 ID 불변 규칙

- **ID는 절대 변경하지 않는다.** 의미가 바뀌면 기존 ID를 `DEPRECATED` 표기하고 새 ID를 할당한다.
- **ID 번호는 재사용하지 않는다.** INV-030이 폐기되어도 INV-030은 영구 사용 불가.
- **새 ID 할당 전 충돌 검사가 필수다.** (섹션 9.1 참조)

---

## 3. 현재 상태와 마이그레이션 계획

### 3.1 현재 상태

| 항목 | 상태 | 상세 |
|------|------|------|
| spec/ | 완성 | 15개 파일 (README 포함 16개), 9,520줄. 모든 요구사항에 ID 부여. 65개 INV 카탈로그화. |
| arch/ | 미분리 | spec/ 내부에 일부 구현 상세가 혼재. 별도 디렉토리 없음. |
| tests/spec/ | 미구축 | INV를 검증하는 전용 테스트 없음. |
| docs/ | 39개 파일 | spec/에 흡수된 레거시, 아키텍처 후보, 가이드 문서 혼재. |

### 3.2 Phase 1: spec/ 내부 normative/non-normative 분리 (PACT 전)

**목표**: spec/ 파일 내에서 규범적(normative) 내용과 비규범적(non-normative) 내용의 경계를 명확히 한다.

**작업 내용**:
1. 각 spec/ 파일의 섹션 7 (Rationale)에 `(non-normative)` 태그 확인 (이미 적용됨)
2. 섹션 3 (Specification) 내부의 "예시", "참고" 등 비규범적 설명에 `> **참고 (non-normative)**:` 블록인용 태그 추가
3. 의사코드가 normative인지 illustrative인지 각각 표기

**기준**:
- `[PREFIX-NNN]`이 붙은 문장 = normative
- `> **참고**:` 블록 = non-normative
- 섹션 7 전체 = non-normative

**완료 기준**: `grep -c 'non-normative' spec/*.md`로 모든 파일에 태그 존재 확인.

### 3.3 Phase 2: arch/ 디렉토리 신설 (PACT 후)

**목표**: spec/에 혼재된 구현 상세를 arch/로 추출한다.

**작업 내용**:
1. `mkdir arch/` 생성
2. spec/ 파일과 1:1 대응하는 arch/ 파일 생성 (같은 파일명)
3. spec/에서 다음 내용을 arch/로 이동:
   - 코드 파일 경로 (`manager/src/policy/*.rs` 등)
   - config 키와 기본값 (`policy.pi.kp = 0.6` 등)
   - CLI 플래그 (`--resilience-config` 등)
   - struct/enum 정의의 코드 레벨 상세
4. spec/에는 의미와 인터페이스만 남김

**arch/ 파일 구조**:
```
arch/
├── 00-overview.md
├── 01-architecture.md
├── 10-protocol.md
├── 11-protocol-messages.md
├── 12-protocol-sequences.md
├── 20-manager.md
├── 21-manager-state.md
├── 22-manager-algorithms.md
├── 23-manager-data.md
├── 30-engine.md
├── 31-engine-state.md
├── 32-engine-algorithms.md
├── 33-engine-data.md
├── 40-cross-cutting.md
└── 41-invariants.md       ← arch 수준의 INV 구현 매핑
```

### 3.4 Phase 3: tests/spec/ 구축 (PACT 후)

**목표**: INV를 자동 검증하는 테스트 모듈 구축.

**작업 순서**:
1. FSM 전이 테이블 테스트 (INV-070~076) -- 전이 테이블이 정형화되어 자동 생성 용이
2. 값 범위 테스트 (INV-031, INV-044, INV-083) -- clamp/범위 검증은 단순하여 즉시 작성 가능
3. 프로토콜 불변식 (INV-020~028) -- 직렬화/역직렬화 round-trip 검증
4. Safety 불변식 (INV-001, INV-005, INV-006) -- 장애 주입, 의존 구조 검증

---

## 4. docs/ 처리

### 4.1 처리 원칙

| 목적지 | 조건 | 처리 |
|--------|------|------|
| **docs/legacy/** | spec/에 내용이 흡수된 문서 | 이동 (삭제 아님) |
| **arch/** | 구현 상세/아키텍처 기술 문서 | Phase 2에서 arch/로 전환 |
| **docs/** (유지) | 빌드 가이드, 실험 가이드, 트러블슈팅 등 운영 문서 | 현 위치 유지 |

### 4.2 파일별 처리 방침 테이블 (39개)

| # | 파일 | 제목 | 목적지 | 사유 |
|---|------|------|--------|------|
| 00 | `00_build_guide.md` | 구현 순서 가이드 | docs/ 유지 | 운영 가이드. spec과 무관. |
| 01 | `01_design_rationale.md` | 설계 결정 및 근거 | arch/ 전환 | 설계 결정 = arch 수준 정보. |
| 02 | `02_core_abstractions.md` | Core 추상화 | docs/legacy/ | spec/30-engine.md + 32-engine-algorithms.md에 흡수. |
| 03 | `03_cpu_backend.md` | CPU 백엔드 | docs/legacy/ | spec/32-engine-algorithms.md, 33-engine-data.md에 흡수. |
| 04 | `04_model_loading.md` | 모델 로딩 파이프라인 | docs/legacy/ | spec/30-engine.md ENG-040~045에 흡수. |
| 05 | `05_tokenizer_and_sampling.md` | 토크나이저 및 샘플링 | docs/legacy/ | spec/32-engine-algorithms.md에 흡수. |
| 06 | `06_opencl_backend.md` | OpenCL Backend | docs/legacy/ | spec/32-engine-algorithms.md, 33-engine-data.md에 흡수. |
| 07 | `07_kernel_implementation.md` | Kernel Implementation | docs/legacy/ | spec/32-engine-algorithms.md에 흡수. |
| 08 | `08_memory_management.md` | Memory Management | docs/legacy/ | spec/33-engine-data.md에 흡수. |
| 09 | `09_attention_mechanism.md` | GPU Attention Mechanism | docs/legacy/ | spec/32-engine-algorithms.md에 흡수. |
| 10 | `10_model_inference.md` | Model Inference Pipeline | docs/legacy/ | spec/30-engine.md, 32-engine-algorithms.md에 흡수. |
| 11 | `11_kv_cache_management.md` | KV Cache 관리 | docs/legacy/ | spec/32-engine-algorithms.md, 33-engine-data.md에 흡수. |
| 12 | `12_hybrid_inference.md` | 하이브리드 추론 | docs/legacy/ | spec/32-engine-algorithms.md에 흡수. |
| 13 | `13_testing_and_benchmarks.md` | 테스트 및 벤치마크 | docs/ 유지 | 운영 가이드. 테스트 실행 방법. |
| 14 | `14_component_status.md` | Component Quality Gates | docs/ 유지 | 운영 상태 추적 문서. |
| 15 | `15_test_strategy.md` | Resilience 테스트 전략 | docs/legacy/ | spec/40-cross-cutting.md, 41-invariants.md에 흡수. |
| 20 | `20_dbus_ipc_spec.md` | D-Bus IPC Specification | docs/legacy/ | **Partially Superseded**. spec/10~12에 흡수. D-Bus 인터페이스명만 참고 가치. |
| 21 | `21_resilience_architecture.md` | Resilience Architecture | docs/legacy/ | spec/20-manager.md, 01-architecture.md에 흡수. |
| 22 | `22_resilience_integration.md` | Resilience Manager 통합 설계 | docs/legacy/ | **Superseded**. spec/30-engine.md CommandExecutor에 흡수. |
| 23 | `23_resilience_test_strategy.md` | Resilience 통합 테스트 전략 | docs/legacy/ | 15_test_strategy.md와 함께 legacy. |
| 24 | `24_resilience_usage_guide.md` | Resilience 사용 가이드 | docs/ 유지 | 운영 가이드. 사용자 대상 실행 방법. |
| 25 | `25_troubleshooting.md` | 트러블슈팅 가이드 | docs/ 유지 | 운영 가이드. |
| 26 | `26_api_reference.md` | Resilience API 레퍼런스 | arch/ 전환 | API 상세 = arch 수준 정보. |
| 27 | `27_manager_architecture.md` | Manager Service Architecture | docs/legacy/ | spec/20-manager.md, 21-manager-state.md에 흡수. |
| 28 | `28_experiment_guide.md` | Resilience 실험 가이드 | docs/ 유지 | 운영 가이드. 실험 실행 방법. |
| 29 | `29_manager_monitor_redesign.md` | Monitor Pattern Redesign | docs/legacy/ | **Superseded**. spec/20-manager.md에 흡수. |
| 30 | `30_evaluation_methodology.md` | KV Cache Eviction 평가 방법론 | docs/ 유지 | 학술 평가 방법론. spec 범위 밖. |
| 31a | `31_memory_architecture.md` | Memory Architecture Overview | docs/legacy/ | spec/33-engine-data.md에 흡수. |
| 31b | `31_perf_comparison_llama_cpp.md` | llm.rs2 vs llama.cpp 비교 | docs/ 유지 | 벤치마크 리포트. 시점 한정 데이터. |
| 32 | `32_kv_offload.md` | KV Cache Offload | docs/legacy/ | spec/32-engine-algorithms.md에 흡수. |
| 34 | `34_profiling_framework_design.md` | Profiling Framework 설계 | arch/ 전환 | 구현 상세 = arch 수준 정보. |
| 35 | `35_experiment_runner_guide.md` | 실험 실행 가이드 | docs/ 유지 | 운영 가이드. 에이전트 인수인계 문서. |
| 36 | `36_policy_design.md` | Hierarchical Policy Design | docs/legacy/ | **Superseded** (파일 내 표기). spec/20~22에 흡수. |
| 37 | `37_protocol_design.md` | Protocol Design | docs/legacy/ | spec/10~12에 흡수. 원래 설계 문서. |
| 38 | `38_eval_refactoring.md` | eval 루프 리팩토링 설계 | arch/ 전환 | 구현 상세 리팩토링 계획. |
| 39 | `39_eval_ll_ratio_budget.md` | eval-ll 비율 기반 KV 버짓 | arch/ 전환 | 구현 상세. |
| 40 | `40_gemma3_support.md` | Gemma 3 1B 지원 설계 | arch/ 전환 | 모델별 구현 상세. |
| - | `PROJECT_CONTEXT.md` | Project Context | docs/ 유지 | 프로젝트 온보딩 문서. |
| - | `README.md` | Documentation Index | docs/ 유지 | 인덱스 (legacy 이동 후 갱신 필요). |

**요약 통계**:

| 목적지 | 파일 수 |
|--------|---------|
| docs/ 유지 | 11 |
| docs/legacy/ | 22 |
| arch/ 전환 | 6 |
| **합계** | **39** |

### 4.3 마이그레이션 실행 명령

```bash
# Phase 2 실행 시 (PACT 후)

# 1. legacy 디렉토리 생성 및 이동 (22개 파일)
mkdir -p docs/legacy
legacy_files=(
  02_core_abstractions.md
  03_cpu_backend.md
  04_model_loading.md
  05_tokenizer_and_sampling.md
  06_opencl_backend.md
  07_kernel_implementation.md
  08_memory_management.md
  09_attention_mechanism.md
  10_model_inference.md
  11_kv_cache_management.md
  12_hybrid_inference.md
  15_test_strategy.md
  20_dbus_ipc_spec.md
  21_resilience_architecture.md
  22_resilience_integration.md
  23_resilience_test_strategy.md
  27_manager_architecture.md
  29_manager_monitor_redesign.md
  31_memory_architecture.md
  32_kv_offload.md
  36_policy_design.md
  37_protocol_design.md
)
for f in "${legacy_files[@]}"; do
  git mv "docs/$f" docs/legacy/
done

# 2. arch/ 디렉토리 생성 및 전환 대상 복사 (6개 파일)
mkdir -p arch
arch_files=(
  01_design_rationale.md
  26_api_reference.md
  34_profiling_framework_design.md
  38_eval_refactoring.md
  39_eval_ll_ratio_budget.md
  40_gemma3_support.md
)
for f in "${arch_files[@]}"; do
  cp "docs/$f" arch/
done
# arch/ 파일을 spec/ 파일명에 맞게 리네임 (수동)

# 3. docs/README.md 갱신
```

---

## 5. spec/ 파일 구조와 ID 체계

### 5.1 파일 목록

현재 spec/ 디렉토리는 15개 정식 파일 + README로 구성된다.

| 파일 | 주요 ID 접두사 | 요구사항 범위 | INV 범위 | CON 범위 | 줄 수 |
|------|---------------|-------------|---------|---------|------|
| `00-overview.md` | SYS | SYS-001~065 | INV-001~006 | - | 434 |
| `01-architecture.md` | SYS | SYS-070~099 | INV-010~018 | - | 674 |
| `10-protocol.md` | PROTO | PROTO-010~075 | INV-020~024 | CON-010~012 | 380 |
| `11-protocol-messages.md` | MSG | MSG-010~100 | INV-025~028 | CON-020~022 | 697 |
| `12-protocol-sequences.md` | SEQ | SEQ-010~104 | - | CON-030~033 | 872 |
| `20-manager.md` | MGR | MGR-010~048 | - | MGR-C01~C02 | 369 |
| `21-manager-state.md` | MGR | MGR-050~087 | - | MGR-C03~C05 | 494 |
| `22-manager-algorithms.md` | MGR-ALG | MGR-ALG-010~072 | INV-030~051 | MGR-C06~C10 | 1,250 |
| `23-manager-data.md` | MGR-DAT | MGR-DAT-010~060 | - | MGR-DAT-C01~C02 | 716 |
| `30-engine.md` | ENG | ENG-010~073 | INV-060~065 | - | 540 |
| `31-engine-state.md` | ENG-ST | ENG-ST-010~070 | INV-070~076 | - | 587 |
| `32-engine-algorithms.md` | ENG-ALG | ENG-ALG-010~095 | - | ENG-ALG-C01~C08 | 990 |
| `33-engine-data.md` | ENG-DAT | ENG-DAT-010~080 | - | ENG-DAT-C01~C07 | 727 |
| `40-cross-cutting.md` | CROSS | CROSS-010~092 | - | CON-040~042 | 397 |
| `41-invariants.md` | INV | INV-001~085 (수집) | INV-080~085 (신규) | - | 276 |

### 5.2 파일 내부 구조 (필수 섹션)

모든 spec/ 파일은 아래 구조를 따른다:

```markdown
# [제목]

> **TL;DR**: 3-5줄 요약

## 1. Purpose and Scope
## 2. Definitions
## 3. Specification          ← 여기만 normative. RFC 2119 키워드 + 요구사항 ID.
## 4. Alternative Behavior   ← (선택적) 예외 사항, 완화 조건.
## 5. Constraints            ← (선택적) 적합성 기준, 유지보수 규칙.
## 6. Examples               ← non-normative. 구체적 시나리오.
## 7. Rationale (non-normative) ← 설계 근거.
```

---

## 6. Spec 변경 규칙

### 6.1 변경 전 확인 사항

Spec 변경 전 반드시 다음을 확인한다:

1. **영향받는 arch/ 파일 식별**: 같은 파일명의 arch/ 파일이 존재하면 동시 갱신 대상.
2. **영향받는 tests/ 파일 식별**: 변경하는 INV-ID를 `grep -r "INV-030" tests/spec/`로 검색.
3. **ID 충돌 검사**: 새 ID를 추가하면 섹션 9.1의 스크립트 실행.

### 6.2 변경 분류

| 분류 | 정의 | 예시 | 절차 |
|------|------|------|------|
| **Normative 변경** | `[PREFIX-NNN]` 요구사항의 의미 변경, 추가, 삭제 | INV-030 범위 변경, 새 INV-086 추가 | 버전 변경. arch/ + tests/ 동시 갱신. |
| **Non-normative 변경** | Rationale 수정, 예시 추가, 오타 수정 | 섹션 7 설명 보강, 예시 코드 수정 | 자유. arch/ + tests/ 갱신 불필요. |

### 6.3 새 요구사항 ID 추가 절차

```bash
# 1. 기존 ID 확인 (예: MGR-ALG에 새 ID 추가하려면)
grep -oE '\[MGR-ALG-[0-9]+\]' spec/22-manager-algorithms.md | sort -t'-' -k3 -n | tail -5

# 2. 마지막 번호 확인 후 +1로 할당
# 예: MGR-ALG-072가 마지막이면 MGR-ALG-073 할당

# 3. 충돌 검사
grep -r 'MGR-ALG-073' spec/
# 결과가 없으면 안전

# 4. spec/에 ID 추가 후 41-invariants.md 동기화 (INV인 경우)
```

### 6.4 새 INV-ID 추가 시 필수 동기화

1. 해당 spec/ 파일에 `[INV-NNN]` 정의 추가.
2. `41-invariants.md` 카탈로그 테이블에 행 추가. 카테고리(Safety/Correctness/Performance/Compatibility)와 검증 방법(static/runtime/test) 명시.
3. (arch/ 존재 시) 대응하는 arch/ 파일에 구현 위치 기술.
4. (tests/spec/ 존재 시) 테스트 함수 추가 또는 COVERAGE.md에 미구현(⬜) 등록.

### 6.5 INV 폐기 절차

```markdown
<!-- 41-invariants.md에서 -->
| INV-030 | ~~22-manager-algorithms ALG-012~~ | ~~can_act=false일 때 integral 미변경~~ | ~~Correctness~~ | ~~runtime, test~~ | **DEPRECATED v2.1** |
```

- 행을 삭제하지 않는다. 취소선 + `DEPRECATED` 표기.
- 번호를 재사용하지 않는다.

---

## 7. Architecture 문서 규칙

> **참고**: arch/ 디렉토리는 Phase 2 (PACT 후)에 생성된다. 이 섹션은 생성 시 따를 규칙을 미리 정의한다.

### 7.1 spec/과 1:1 대응

- arch/ 파일명 = spec/ 파일명과 동일 (예: `arch/22-manager-algorithms.md`)
- spec/ 파일이 없는 arch/ 파일은 존재할 수 없다.
- arch/ 파일에 독자적인 요구사항 ID를 만들지 않는다. **ID 원천은 항상 spec/**.

### 7.2 arch/ 파일에 포함되는 내용

| 항목 | 예시 |
|------|------|
| 코드 파일 경로 | `manager/src/policy/pi_controller.rs` |
| 구조체/함수 매핑 | `INV-031 → PiController::update():87` |
| config 키와 기본값 | `policy.pi.integral_clamp = 2.0` |
| CLI 플래그 | `--resilience-config <path>` |
| 빌드 feature gate | `#[cfg(feature = "resilience")]` |
| 외부 의존성 버전 | `serde_json = "1.0"` |

### 7.3 arch/ 파일 템플릿

```markdown
# [제목] -- Architecture

> spec/XX-YYYY.md의 구현 상세.

## 코드 매핑

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| INV-031  | manager/src/policy/pi_controller.rs:87 | f64::clamp | |
| MGR-ALG-012 | manager/src/policy/pi_controller.rs:80-95 | update() 메서드 | |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| policy.pi.integral_clamp | f64 | 2.0 | INV-031 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| --resilience-config | 정책 설정 파일 경로 | MGR-DAT-010 |
```

### 7.4 갱신 의무

코드 변경 시 관련 arch/ 파일을 반드시 갱신한다. 특히:
- 파일 경로 변경 (rename, 디렉토리 이동)
- config 키 변경
- CLI 플래그 추가/변경
- struct 필드 추가/삭제

---

## 8. 테스트 규칙

### 8.1 테스트 디렉토리 구조

```
tests/
└── spec/
    ├── mod.rs
    ├── test_inv_030.rs      ← INV-030 검증
    ├── test_inv_031.rs      ← INV-031 검증
    ├── test_fsm_operating_mode.rs  ← OperatingMode FSM 전체 전이
    ├── test_fsm_engine_state.rs    ← EngineState FSM 전체 전이
    └── helpers/
        └── mod.rs           ← 공유 테스트 유틸리티
```

### 8.2 파일명 규칙

| 패턴 | 용도 | 예시 |
|------|------|------|
| `test_inv_NNN.rs` | 개별 INV 검증 | `test_inv_030.rs` |
| `test_inv_NNN_NNN.rs` | 연속 INV 묶음 검증 | `test_inv_020_024.rs` (프로토콜 INV 묶음) |
| `test_fsm_<name>.rs` | FSM 전이 테이블 전수 검증 | `test_fsm_operating_mode.rs` |
| `test_proto_<name>.rs` | 프로토콜 round-trip 검증 | `test_proto_roundtrip.rs` |

### 8.3 함수명 규칙

테스트 함수명에 반드시 INV-ID를 포함한다:

```rust
// Good
#[test]
fn test_inv_030_can_act_false_integral_unchanged() { ... }

#[test]
fn test_inv_031_integral_clamped_to_range() { ... }

#[test]
fn test_inv_072_suspend_overrides_all() { ... }

// Bad -- INV-ID 없음
#[test]
fn test_pi_controller_integral() { ... }
```

### 8.4 테스트 전략 3단계

INV마다 다음 3단계를 순서대로 검토한다:

**단계 1: 자동화 가능 -- 단위 테스트**

대부분의 INV가 여기에 해당한다. 값 범위, 상태 전이, 수식 결과 등.

```rust
#[test]
fn test_inv_031_integral_always_in_range() {
    let mut pi = PiController::new(Config { integral_clamp: 2.0, .. });
    for _ in 0..10000 {
        pi.update(1.0, 0.01); // 극단적 입력
        assert!(pi.integral >= 0.0 && pi.integral <= 2.0,
            "INV-031 violation: integral={}", pi.integral);
    }
}
```

**단계 2: E2E/타이밍 필요 -- 기능 분리하여 단위 테스트로**

타이밍, IPC 등이 필요한 경우 핵심 로직을 분리하여 단위 테스트한다.

```rust
// INV-064: heartbeat_interval 내 최소 1회 Heartbeat 전송
// E2E 테스트 대신, poll()이 Heartbeat를 생성하는 로직을 단위 테스트
#[test]
fn test_inv_064_poll_generates_heartbeat() {
    let mut executor = CommandExecutor::new(..);
    executor.set_elapsed(Duration::from_secs(2)); // interval 초과
    let plan = executor.poll();
    assert!(plan.heartbeat.is_some(), "INV-064: heartbeat 미생성");
}
```

**단계 3: 불가 -- 제약사항으로 명시**

자동 테스트가 근본적으로 불가능한 경우 (예: `static` 검증만 가능한 아키텍처 제약):

```markdown
<!-- COVERAGE.md에 기록 -->
| INV-001 | 🔶 제약사항 | static 검증 전용. Cargo.toml 의존 구조로만 보장. CI dependency graph 검사로 대체. |
```

### 8.5 FSM 테스트: 전이 테이블 전수 검증

FSM 전이 테이블의 모든 (상태 x 입력) 조합을 테스트한다.

```rust
// test_fsm_operating_mode.rs
// 21-manager-state.md의 전이 테이블에서 자동 생성

#[test]
fn test_fsm_operating_mode_all_transitions() {
    // 전이 테이블 정의 (spec/21-manager-state.md에서 추출)
    let transitions = vec![
        // (현재상태, 입력, 기대상태)
        (Normal,   PeakAboveCritical,  Critical),
        (Normal,   PeakAboveWarning,   Warning),
        (Normal,   PeakBelowWarning,   Normal),
        (Warning,  PeakAboveCritical,  Critical),  // INV-032: 즉시 에스컬레이션
        (Warning,  PeakBelowRelease,   Normal),     // INV-033: 1단계 디에스컬레이션
        (Critical, PeakBelowRelease,   Warning),    // INV-033: 1단계씩만
        // ... 전체 조합
    ];

    for (current, input, expected) in transitions {
        let result = OperatingMode::next_mode(current, input);
        assert_eq!(result, expected,
            "FSM violation: {:?} + {:?} -> {:?} (expected {:?})",
            current, input, result, expected);
    }
}
```

### 8.6 프로퍼티 기반 테스트

범위 불변식은 프로퍼티 테스트로 강화한다:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_inv_044_parametrize_output_range(
        pressure in 0.0f64..=1.0,
        range_min in 0.0f64..=0.5,
        range_max in 0.5f64..=1.0,
    ) {
        let value = parametrize(pressure, range_min, range_max);
        prop_assert!(value >= range_min && value <= range_max,
            "INV-044 violation: value={}, range=[{}, {}]", value, range_min, range_max);
    }
}
```

---

## 9. 일관성 검사

### 9.1 자동 검사 (CI / pre-commit)

#### scripts/check_id_collision.sh

새 ID가 기존 ID와 충돌하지 않는지 확인한다.

```bash
#!/usr/bin/env bash
# scripts/check_id_collision.sh
# 사용법: ./scripts/check_id_collision.sh
# 종료코드: 0 = 충돌 없음, 1 = 충돌 발견

set -euo pipefail

SPEC_DIR="spec"
DUPLICATES=$(
  grep -roE '\[[A-Z]+-[A-Z]*-?[0-9]+[a-z]?\]' "$SPEC_DIR"/*.md \
  | sed 's/.*://' \
  | sort \
  | uniq -d
)

if [ -n "$DUPLICATES" ]; then
  echo "ERROR: 중복 ID 발견:"
  echo "$DUPLICATES"
  for id in $DUPLICATES; do
    echo "  $id 위치:"
    grep -rn "$id" "$SPEC_DIR"/*.md | head -5
  done
  exit 1
fi

echo "OK: ID 충돌 없음."
exit 0
```

#### scripts/check_spec_coverage.sh

spec/의 모든 INV-ID가 tests/spec/에 대응되는지 확인한다.

```bash
#!/usr/bin/env bash
# scripts/check_spec_coverage.sh
# 사용법: ./scripts/check_spec_coverage.sh
# 종료코드: 0 = 전수 커버, 1 = 미커버 INV 존재

set -euo pipefail

SPEC_DIR="spec"
TEST_DIR="tests/spec"

# 1. spec/에서 모든 INV-ID 추출
SPEC_INVS=$(grep -roE 'INV-[0-9]+' "$SPEC_DIR"/41-invariants.md \
  | sed 's/.*://' | sort -u)

# 2. tests/spec/에서 참조된 INV-ID 추출
if [ -d "$TEST_DIR" ]; then
  TEST_INVS=$(grep -roE 'inv_[0-9]+' "$TEST_DIR"/ \
    | sed 's/.*inv_/INV-/' | sort -u)
else
  TEST_INVS=""
fi

# 3. spec에는 있지만 tests에 없는 INV 찾기
MISSING=$(comm -23 <(echo "$SPEC_INVS") <(echo "$TEST_INVS"))

if [ -n "$MISSING" ]; then
  MISSING_COUNT=$(echo "$MISSING" | wc -l | tr -d ' ')
  TOTAL_COUNT=$(echo "$SPEC_INVS" | wc -l | tr -d ' ')
  echo "WARNING: $MISSING_COUNT / $TOTAL_COUNT INV가 테스트에 미대응:"
  echo "$MISSING"
  exit 1
fi

echo "OK: 모든 INV가 테스트에 대응됨."
exit 0
```

#### 관련 INV 변경 시 자동 테스트 실행

`.cargo/config.toml` 또는 CI 설정에서 INV 관련 코드 변경 시 해당 테스트를 자동 실행한다:

```bash
# 변경된 INV 관련 테스트만 실행하는 예시
# CI에서: 변경된 파일에서 INV-NNN 추출 → 해당 test_inv_NNN 실행
CHANGED_INVS=$(git diff --name-only HEAD~1 | xargs grep -ohE 'INV-[0-9]+' 2>/dev/null | sort -u)
for inv in $CHANGED_INVS; do
  NUM=$(echo "$inv" | grep -oE '[0-9]+')
  cargo test "test_inv_${NUM}" --test spec 2>/dev/null || true
done
```

### 9.2 수동 검사 (이벤트 기반)

| 트리거 | 검사 범위 | 절차 |
|--------|----------|------|
| Spec normative 변경 | 영향받는 arch/ + tests/ 전체 | 변경 ID를 `grep -r` 으로 3계층 검색. 불일치 수정. |
| Major 리팩토링 | 전체 arch/ + tests/ | `check_spec_coverage.sh` + `check_id_collision.sh` + 수동 교차검증. |
| 릴리스 | 전체 INV 테스트 | `cargo test --test spec` 전체 실행. VERIFICATION.md 기록. |

---

## 10. VERIFICATION.md 형식

검사 이력 로그. `spec/VERIFICATION.md`에 기록한다.

```markdown
# Verification Log

## 2026-03-28: Phase 1 전수 검사

| 항목 | 값 |
|------|-----|
| **날짜** | 2026-03-28 |
| **범위** | INV-001 ~ INV-085 전체 (65개) |
| **트리거** | Phase 3 테스트 구축 시작 |
| **결과** | 52/65 PASS, 0 FAIL, 13 미구현 |
| **커버리지** | 80% (52/65) |
| **발견** | INV-051 (동시 적용 relief 귀속) 테스트 불가 → 제약사항 전환 |
| **해결** | INV-051을 COVERAGE.md에 🔶 제약사항으로 등록 |

---

## 2026-04-15: Protocol INV 추가 검증

| 항목 | 값 |
|------|-----|
| **날짜** | 2026-04-15 |
| **범위** | INV-020 ~ INV-028 (9개) |
| **트리거** | 프로토콜 v2 변경 |
| **결과** | 9/9 PASS |
| **커버리지** | 100% (9/9) |
| **발견** | 없음 |
| **해결** | - |
```

---

## 11. COVERAGE.md 형식

INV별 테스트 상태 매트릭스. `spec/COVERAGE.md`에 기록한다.

**상태 기호**:

| 기호 | 의미 |
|------|------|
| ✅ | PASS -- 테스트 존재 + 통과 |
| ❌ | FAIL -- 테스트 존재 + 실패 |
| ⬜ | 미구현 -- 테스트 미작성 |
| 🔶 | 제약사항 -- 자동 테스트 불가. 사유 명시. |

### 전체 INV 커버리지 매트릭스

```markdown
# INV Coverage Matrix

> 최종 갱신: 2026-03-28
> 전체: 65개 | ✅ 0 | ❌ 0 | ⬜ 49 | 🔶 16

## System/Architecture [INV-001 ~ INV-018]

| INV | 요약 | 카테고리 | 검증 | 상태 | 테스트/사유 |
|-----|------|---------|------|------|-----------|
| INV-001 | 2 독립 프로세스, 직접 의존 금지 | Safety | static | 🔶 | Cargo.toml 의존 구조 검증. CI graph 검사. |
| INV-002 | NEON은 ARM64에서만 | Safety | static | 🔶 | #[cfg(target_arch)] 컴파일 타임 보장. |
| INV-003 | 미지원 architectures 로딩 거부 | Correctness | runtime | ⬜ | |
| INV-004 | QCF 수집 시 lossy action → QcfMetric 필수 | Correctness | test | ⬜ | |
| INV-005 | Manager 장애 → Engine 미중단 | Safety | test | ⬜ | |
| INV-006 | Engine 장애 → Manager 미중단 | Safety | test | ⬜ | |
| INV-010 | Engine-Manager 직접 의존 금지 | Safety | static | 🔶 | Cargo.toml. INV-001 재확인. |
| INV-011 | Shared → Engine/Manager 미의존 | Safety | static | 🔶 | Cargo.toml. |
| INV-012 | Backend trait = 유일한 HW 추상화점 | Correctness | static | 🔶 | 코드 리뷰. |
| INV-013 | Monitor 스레드 장애 미전파 | Safety | static, test | 🔶 | 아키텍처 제약. |
| INV-014 | seq_id 단조 증가 | Correctness | runtime | ⬜ | |
| INV-015 | Capability 세션당 1회 | Correctness | runtime, test | ⬜ | |
| INV-016 | 배타 그룹 동시 활성화 금지 | Correctness | runtime, test | ⬜ | |
| INV-017 | QCF+lossy → QcfMetric 필수 (재확인) | Correctness | test | ⬜ | => INV-004 |
| INV-018 | 추론 루프 단일 스레드 | Safety | static | 🔶 | 아키텍처 제약. |

## Protocol [INV-020 ~ INV-028]

| INV | 요약 | 카테고리 | 검증 | 상태 | 테스트/사유 |
|-----|------|---------|------|------|-----------|
| INV-020 | seq_id 단조 증가 | Correctness | runtime | ⬜ | |
| INV-021 | seq_id 재사용 금지 | Correctness | runtime | ⬜ | |
| INV-022 | Directive → 정확히 1 Response | Correctness | runtime, test | ⬜ | |
| INV-023 | Response.seq_id == Directive.seq_id | Correctness | runtime | ⬜ | |
| INV-024 | len(results) == len(commands) | Correctness | runtime | ⬜ | |
| INV-025 | len(results) == len(commands) (재확인) | Correctness | runtime | ⬜ | => INV-024 |
| INV-026 | 수신 seq_id에만 Response | Correctness | runtime | ⬜ | |
| INV-027 | serde 변경 = 프로토콜 버전 변경 | Compatibility | static | 🔶 | 코드 리뷰. |
| INV-028 | 새 필드 → serde(default) 필수 | Compatibility | static | 🔶 | 코드 리뷰. |

## Manager Algorithm [INV-030 ~ INV-051]

| INV | 요약 | 카테고리 | 검증 | 상태 | 테스트/사유 |
|-----|------|---------|------|------|-----------|
| INV-030 | can_act=false → integral 미변경 | Correctness | runtime, test | ⬜ | |
| INV-031 | integral in [0, integral_clamp] | Correctness | runtime | ⬜ | |
| INV-032 | 에스컬레이션 즉시 (직행 가능) | Correctness | test | ⬜ | |
| INV-033 | 디에스컬레이션 1단계씩 | Correctness | test | ⬜ | |
| INV-034 | warning_release < warning_threshold | Correctness | runtime | ⬜ | |
| INV-035 | critical_release < critical_threshold | Correctness | runtime | ⬜ | |
| INV-036 | warning_threshold < critical_threshold | Correctness | runtime | ⬜ | |
| INV-037 | Warning에서 Lossy 선택 금지 | Correctness | runtime, test | ⬜ | |
| INV-038 | 활성 중 액션 재선택 금지 | Correctness | runtime | ⬜ | |
| INV-039 | Lossless cost = 0 | Correctness | runtime | ⬜ | |
| INV-040 | QCF 없는 Lossy = INFINITY cost | Correctness | runtime | ⬜ | |
| INV-041 | 배타 그룹 동시 미포함 (재확인) | Correctness | runtime, test | ⬜ | => INV-016 |
| INV-042 | latency 초과 조합 배제 | Performance | runtime | ⬜ | |
| INV-043 | 완전 해소 > best-effort | Correctness | runtime | ⬜ | |
| INV-044 | parametrize 출력 [min, max] | Correctness | runtime | ⬜ | |
| INV-045 | primary_domain 매핑 고정 | Correctness | static | 🔶 | 코드 구조. |
| INV-046 | RLS gain vector k = f(P, phi) | Correctness | test | ⬜ | |
| INV-047 | bias = EMA(lr=0.1) | Correctness | test | ⬜ | |
| INV-048 | P matrix: D x D 대칭 양정치, 초기 100*I | Correctness | runtime | ⬜ | |
| INV-049 | lambda in (0, 1] | Correctness | runtime | ⬜ | |
| INV-050 | 관찰 relief latency = 0.0 | Correctness | runtime | ⬜ | |
| INV-051 | 동시 적용 시 개별 분리 불가 | Correctness | runtime | 🔶 | 설계 한계. 분리 불가는 시스템 제약. |

## Engine Architecture [INV-060 ~ INV-065]

| INV | 요약 | 카테고리 | 검증 | 상태 | 테스트/사유 |
|-----|------|---------|------|------|-----------|
| INV-060 | poll() 토큰당 최대 1회 | Performance | static | 🔶 | 코드 구조. |
| INV-061 | ExecutionPlan 1회성 | Safety | static | 🔶 | 코드 구조. |
| INV-062 | Suspend 시 evict/switch/prepare = None | Safety | runtime | ⬜ | |
| INV-063 | MessageLoop = Transport 유일 소유자 | Safety | static | 🔶 | ownership 타입 시스템. |
| INV-064 | heartbeat_interval 내 Heartbeat 전송 | Correctness | runtime | ⬜ | |
| INV-065 | Backend = Send + Sync | Safety | static | 🔶 | trait bound. |

## Engine State Machine [INV-070 ~ INV-076]

| INV | 요약 | 카테고리 | 검증 | 상태 | 테스트/사유 |
|-----|------|---------|------|------|-----------|
| INV-070 | from_levels() 순수 함수 | Correctness | static | 🔶 | 함수 시그니처. |
| INV-071 | EngineState 전이 CommandExecutor 내부만 | Correctness | static | ⬜ | |
| INV-072 | Suspend 존재 시 반환=[Suspend] | Safety | runtime, test | ⬜ | |
| INV-073 | RestoreDefaults 조건부 반환 | Correctness | runtime, test | ⬜ | |
| INV-074 | suspended → evict/switch/prepare=None | Safety | runtime | ⬜ | => INV-062 |
| INV-075 | Resume → Normal, throttle 0 | Correctness | runtime | ⬜ | |
| INV-076 | RestoreDefaults → 전체 초기화 | Correctness | runtime | ⬜ | |

## Cross-cutting [INV-080 ~ INV-085]

| INV | 요약 | 카테고리 | 검증 | 상태 | 테스트/사유 |
|-----|------|---------|------|------|-----------|
| INV-080 | async 런타임 금지 | Safety | static | 🔶 | Cargo.toml. |
| INV-081 | IPC = JSON 전용 | Compatibility | static | ⬜ | |
| INV-082 | 1:1 단일 클라이언트 | Safety | runtime | ⬜ | |
| INV-083 | PI output [0, 1] | Correctness | runtime | ⬜ | |
| INV-084 | ActionSelector stateless, predict 읽기 전용 | Correctness | static | 🔶 | 코드 구조. |
| INV-085 | Normal 모드 액션 미발행 | Correctness | runtime, test | ⬜ | |
```

---

## 12. 자주 묻는 질문 (FAQ)

### Q1: "spec에 있는데 코드에 없는 것은?"

**A**: 코드 구현이 필요하다. spec/이 설계 기준(source of truth)이다. spec/에 정의된 요구사항은 구현 의무가 있다. 즉시 구현이 어려우면 arch/ 파일에 `TODO: 미구현` 표기하고 이슈를 등록한다.

### Q2: "코드에 있는데 spec에 없는 것은?"

**A**: 두 가지 중 하나를 판단한다:
1. **spec에 추가해야 한다** -- 코드의 동작이 설계 의도에 부합하며, 명세화할 가치가 있는 경우.
2. **코드를 제거해야 한다** -- dead code이거나 spec과 모순되는 경우.

판단이 어려우면 아키텍트에게 문의한다. 임의로 코드를 남겨두지 않는다.

### Q3: "arch/를 먼저 만들어야 테스트를 만들 수 있나?"

**A**: 아니오. **spec/만으로 테스트 작성이 가능하다.** spec/은 구현 독립적인 WHAT을 정의하므로, 코드 경로를 몰라도 불변식의 입출력 조건으로 테스트를 작성할 수 있다. arch/는 편의 참조일 뿐 테스트의 전제 조건이 아니다.

```rust
// spec/22-manager-algorithms.md의 INV-031만 보고 작성 가능:
// "integral은 항상 [0, integral_clamp] 범위 내이다"
#[test]
fn test_inv_031_integral_clamped() {
    let mut pi = PiController::new(..);
    // 어떤 입력이든 integral이 범위를 벗어나면 안 된다
}
```

### Q4: "INV-ID를 변경하면?"

**A**: **절대 변경 금지.** INV-ID는 불변 식별자이다.
- 의미가 바뀌면: 기존 ID를 `DEPRECATED` 처리하고 새 ID를 할당한다.
- 오타가 있으면: 내용만 수정하고 ID는 유지한다.
- 폐기하면: `DEPRECATED` 표기. 번호 재사용 금지.

### Q5: "같은 INV가 여러 spec/ 파일에 등장하면?"

**A**: **재확인(restatement) 관계**이다. 원본과 사본이 있으며, 원본이 정본이다. 41-invariants.md의 "재확인 관계" 섹션에 명시되어 있다:

| INV | 원본 | 재확인 |
|-----|------|--------|
| INV-017 | INV-004 | 01-architecture에서 00-overview 재확인 |
| INV-025 | INV-024 | 11-messages에서 10-protocol 재확인 |
| INV-041 | INV-016 | 22-algorithms에서 01-architecture 재확인 |
| INV-062 | INV-074 | 30-engine과 31-engine-state 교차 재확인 |

원본이 변경되면 재확인 INV도 동기화해야 한다.

### Q6: "spec/ 변경 없이 코드만 리팩토링하면?"

**A**: spec/ 변경 불필요. arch/ 파일의 코드 경로만 갱신한다. tests/spec/은 spec/ 기준이므로 코드 리팩토링으로 인한 테스트 변경은 구현 세부(import 경로 등)에 한정된다.

### Q7: "새 컴포넌트/알고리즘을 추가하면?"

**A**: 다음 순서를 따른다:
1. spec/ 파일에 요구사항 추가 (ID 할당, 충돌 검사)
2. INV가 있으면 41-invariants.md 카탈로그 갱신
3. arch/ 파일에 구현 상세 기술 (Phase 2 이후)
4. tests/spec/에 테스트 추가 (Phase 3 이후)
5. COVERAGE.md 갱신

### Q8: "normative 변경 vs non-normative 변경을 어떻게 구분하나?"

**A**: `[PREFIX-NNN]` 태그가 붙은 문장을 변경하면 normative 변경이다. 그 외는 non-normative이다.

```markdown
<!-- normative (변경 시 버전 변경 필요) -->
[INV-031] integral은 항상 [0, integral_clamp] 범위 내이다. (MUST)

<!-- non-normative (자유 변경) -->
> **참고**: integral_clamp의 기본값 2.0은 실험적으로 결정되었다.
```

---

## 부록 A: 빠른 참조 명령어

```bash
# 특정 ID가 어디에 정의되어 있는지 찾기
grep -rn 'INV-031' spec/

# 특정 ID의 3계층 추적
grep -rn 'INV-031' spec/ arch/ tests/spec/

# 전체 INV 목록 추출 (41-invariants.md에서)
grep -oE 'INV-[0-9]+' spec/41-invariants.md | sort -t'-' -k2 -n -u

# ID 충돌 검사
./scripts/check_id_collision.sh

# 테스트 커버리지 검사
./scripts/check_spec_coverage.sh

# 특정 INV 관련 테스트만 실행
cargo test "test_inv_031" --test spec

# FSM 테스트 전체 실행
cargo test "test_fsm" --test spec

# 변경된 INV 확인 (최근 커밋 기준)
git diff HEAD~1 -- spec/ | grep -oE 'INV-[0-9]+' | sort -u
```

## 부록 B: 체크리스트

### Spec 변경 시 체크리스트

- [ ] 변경 대상이 normative인지 non-normative인지 확인
- [ ] 새 ID 추가 시 `check_id_collision.sh` 실행
- [ ] 새 INV 추가 시 `41-invariants.md` 카탈로그 갱신
- [ ] 영향받는 arch/ 파일 식별 및 갱신 (존재 시)
- [ ] 영향받는 tests/spec/ 파일 식별 및 갱신 (존재 시)
- [ ] COVERAGE.md 갱신 (INV 추가/변경 시)
- [ ] 재확인 관계 확인 (변경한 INV가 다른 파일에서 재확인되는지)

### 코드 변경 시 체크리스트

- [ ] 변경이 기존 INV를 위반하지 않는지 확인
- [ ] arch/ 파일의 코드 경로 갱신 (파일 이동/리네임 시)
- [ ] 관련 INV 테스트 실행 (`cargo test "test_inv_NNN"`)

### 릴리스 시 체크리스트

- [ ] `cargo test --test spec` 전체 실행
- [ ] `check_spec_coverage.sh` 실행
- [ ] `check_id_collision.sh` 실행
- [ ] VERIFICATION.md에 검사 이력 기록
- [ ] COVERAGE.md 최신화
