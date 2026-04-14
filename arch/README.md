# llm_rs2 Architecture Documents

이 디렉토리는 spec/의 구현 상세를 기술한다. spec/과 1:1 대응하는 파일 구조를 따른다.

## 관계

```
spec/  → 불변. WHAT (무엇을 보장해야 하는가)
arch/  → 가변. HOW (코드에서 어떻게 구현하는가)  ← 이 디렉토리
```

## 파일 구조

| arch/ 파일 | 대응 spec/ | 내용 |
|-----------|-----------|------|
| `00-overview.md` | `spec/00-overview.md` | 크레이트 구조, 빌드 프로파일, Feature Gate |
| `01-architecture.md` | `spec/01-architecture.md` | 서브시스템-모듈 매핑, Transport 구현체 |
| `10-protocol.md` | `spec/10-protocol.md` | Wire format 구현, MAX_PAYLOAD_SIZE |
| `11-protocol-messages.md` | `spec/11-protocol-messages.md` | 메시지 타입 serde 매핑 |
| `12-protocol-sequences.md` | `spec/12-protocol-sequences.md` | 시퀀스 구현 진입점 |
| `20-manager.md` | `spec/20-manager.md` | Monitor/Policy/Emitter 코드 매핑 |
| `21-manager-state.md` | `spec/21-manager-state.md` | FSM 코드 위치, ThresholdEvaluator |
| `22-manager-algorithms.md` | `spec/22-manager-algorithms.md` | PI Controller, Supervisory, ActionSelector |
| `23-manager-data.md` | `spec/23-manager-data.md` | TOML 스키마, Config 키 |
| `30-engine.md` | `spec/30-engine.md` | 서브시스템 모듈, CLI 플래그, Transport |
| `31-engine-state.md` | `spec/31-engine-state.md` | FSM 코드, ExecutionPlan, EngineCommand |
| `32-engine-algorithms.md` | `spec/32-engine-algorithms.md` | Eviction, KIVI, QCF, Pipeline |
| `33-engine-data.md` | `spec/33-engine-data.md` | Trait 구현체, Buffer, DType |
| `40-cross-cutting.md` | `spec/40-cross-cutting.md` | Fail-safety, 로깅, 에러 전파 |
| `41-invariants.md` | `spec/41-invariants.md` | 65개 INV 구현 위치 마스터 테이블 |
| `50-test-tools.md` | `spec/50-test-tools.md` | mock_engine, mock_manager 구현 매핑 |

## 독립 설계 문서

spec/ 대응이 아닌 독립적인 feature 설계 문서:

| 파일 | 내용 |
|------|------|
| `tensor_partition.md` | CPU-GPU Cooperative Inference (Option B) — Tensor Partition 설계 |
| `cpu_flash_decoding.md` | CPU Attention KV-Split 병렬화 (Step 2 — `attention_gen_f16_neon`) 설계 |
| `clock_abstraction.md` | Manager `Clock` trait 추상화 (테스트 용이성, 시뮬레이터 시간 주입) |

## 규칙

- arch/ 파일에 독자적 요구사항 ID를 만들지 않는다. **ID 원천은 항상 spec/**.
- 코드 변경 시 관련 arch/ 파일을 갱신한다.
- 상세 규칙: `.claude/skills/spec-manage/SKILL.md` 참조.
