# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `engine/src/`, `manager/`, `shared/`

---

## [P0] Experiment 스케줄 JSON 파일 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Experiment Mode 구현 완료 ✅
- **Description**: Round 1~2 실험용 스케줄 JSON 파일 생성. `experiments/configs/` 디렉토리
- **Acceptance Criteria**:
  - baseline.json ✅ (완료)
  - 속도 실험 (128 토큰): thermal_warning_32, thermal_critical_32, thermal_crit_32_recover_96, compute_cpu_32, energy_emergency_32
  - 품질 실험 (512 토큰): memory_warning_256, memory_critical_256, memory_crit_256_recover_384
  - 메모리 실험 (1024 토큰): memory_critical_512, memory_critical_256_long, memory_critical_768, memory_crit_512_recover_768
  - 모든 JSON이 스케줄 스키마 준수
- **Notes**: `experiments/PLAN.md` Section 3.2 + Section 5 참조

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
