# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `engine/src/`, `manager/`, `shared/`

---

*(현재 P0 작업 완료. 다음 실험 실행은 Tester 역할)*

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

## [P0] generate.rs Experiment Mode 구현
- **Status**: DONE
- **Notes**: 커밋 82d7cf2. CLI 플래그 5개, ExperimentSchedule, SystemSampler, JsonlWriter, per-token JSONL, 8 테스트

## [P0] SystemSampler 구현
- **Status**: DONE
- **Notes**: 커밋 82d7cf2. RSS/CPU util/CPU freq/Thermal/GPU 수집, interval 제어, snapshot

## [P0] Experiment 스케줄 JSON 파일 작성
- **Status**: DONE
- **Notes**: 커밋 39e6570. 12개 파일 (baseline + speed 5 + quality 3 + memory 3), 모두 Rust 파싱 검증 완료

## [P1] IPC Transport 추상화 구현
- **Status**: DONE
- **Notes**: 커밋 c2b7c64. SignalTransport trait + DbusTransport + UnixSocketTransport + MockTransport

## [P1] Hybrid + Eviction 통합 구현
- **Status**: DONE
- **Notes**: 커밋 dc78418. CacheManager, EvictionPolicy(none/sliding/h2o) 통합

## [P1] Hybrid + Resilience 통합 구현
- **Status**: DONE
- **Notes**: 커밋 db2df27. SwitchBackend 실제 동작, GPU↔CPU 양방향 전환

## [P2] Manager 서비스 Collector 구현
- **Status**: DONE
- **Notes**: 커밋 bd57980. 4개 Collector, 13개 테스트

## [P2] Manager 서비스 PolicyEngine + Emitter 구현
- **Status**: DONE
- **Notes**: 커밋 89e2a34 + b35803a. OCP PolicyEngine trait, 26개 테스트

</details>
