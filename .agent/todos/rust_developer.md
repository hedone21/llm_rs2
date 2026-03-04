# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `engine/src/`, `manager/`, `shared/`

---

## [P1] IPC Transport 추상화 구현
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 아키텍트의 IPC Transport 설계 완료
- **Description**: `SignalTransport` trait 구현. DbusTransport, UnixSocketTransport, MockTransport
- **Acceptance Criteria**: 3종 transport 구현, 기존 resilience 유닛 테스트 통과
- **Notes**: 커밋 c2b7c64에서 구현 완료. `engine/src/resilience/transport.rs` + `dbus_transport.rs`

## [P1] Hybrid + Eviction 통합 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 아키텍트의 Hybrid 통합 설계
- **Description**: `engine/src/bin/generate_hybrid.rs`에 CacheManager 연동, KV eviction 지원 추가. 현재 hybrid는 max_seq_len 초과 시 추론 중단됨
- **Acceptance Criteria**: hybrid 모드에서 eviction 정상 동작, 장시간 추론 가능
- **Notes**: 기존 `engine/src/bin/generate.rs` eviction 패턴 참고. current로 승격

## [P1] Hybrid + Resilience 통합 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 아키텍트의 Hybrid 통합 설계
- **Description**: `engine/src/bin/generate_hybrid.rs`에 Resilience checkpoint 추가. GPU→CPU 역방향 전환 구현 (Thermal Critical 시)
- **Acceptance Criteria**: hybrid 모드에서 resilience 신호 반응, GPU→CPU 전환 동작
- **Notes**: SwitchBackend 액션이 실제로 동작하는 첫 구현. IPC Transport 완료로 블로커 해소

## [P2] Manager 서비스 Collector 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (workspace 구조 확정 완료)
- **Description**: `manager/src/`에 시스템 리소스 수집기 구현. MemoryCollector(/proc/meminfo, PSI), ThermalCollector(/sys/class/thermal), ComputeCollector(/proc/stat), EnergyCollector(UPower/sysfs)
- **Acceptance Criteria**: 각 Collector가 정확한 시스템 데이터 수집, 유닛 테스트 포함
- **Notes**: Android와 Linux에서 sysfs 경로 차이 고려. workspace 블로커 해소 → current로 승격

## [P2] Manager 서비스 PolicyEngine + Emitter 구현
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Collector 구현
- **Description**: Threshold + Hysteresis 기반 Level 결정 로직, TOML 설정 파싱, DbusEmitter/UnixSocketEmitter 송신 구현
- **Acceptance Criteria**: 정책 엔진이 올바른 level 결정, 히스테리시스 동작, 시그널 송신 확인
- **Notes**: docs/20_dbus_ipc_spec.md의 임계값/히스테리시스 예시 참고
