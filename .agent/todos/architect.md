# Architect TODO

> **역할**: 시스템 설계, 트레이트/인터페이스 정의, 모듈 간 의존성 관리, 기술 결정
> **소유 영역**: `engine/src/core/`, `shared/`, Cargo workspace 구조, `ARCHITECTURE.md`

---

## [P1] IPC Transport 추상화 설계
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `SignalTransport` trait 설계. D-Bus(Linux)/Unix Domain Socket(Android)/Mock(테스트) 3종 전송 계층 추상화
- **Acceptance Criteria**: trait 시그니처, 메시지 포맷 명세, 플랫폼 선택 로직 설계 문서
- **Notes**: 커밋 c2b7c64에서 구현 완료. `engine/src/resilience/transport.rs`에 SignalTransport trait + UnixSocketTransport + MockTransport 존재

## [P2] Cargo workspace 구조 설계 (Manager 서비스)
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: IPC Transport 추상화 설계
- **Description**: llm_rs2와 Manager 서비스 간 signal 타입 공유를 위한 Cargo workspace 구조 설계
- **Acceptance Criteria**: 프로젝트 구조 설계 문서, 의존성 그래프
- **Notes**: 커밋 95af0a3에서 구현 완료. engine/shared/manager 3-crate workspace 구조 확립

## [P1] Hybrid 추론 + Eviction/Resilience 통합 설계
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `engine/src/bin/generate_hybrid.rs`에 CacheManager(eviction), Resilience checkpoint, GPU→CPU 역방향 전환 통합 설계. 현재 hybrid는 eviction/resilience 미지원 상태
- **Acceptance Criteria**: 통합 설계 문서, 역방향 전환 시퀀스, 변경 범위 명세
- **Notes**: 구현 완료. KV 마이그레이션 3중 중복을 `migrate_kv_caches()` 헬퍼로 리팩토링 (-40줄). Evict, SwitchBackend(양방향), Throttle, Suspend 모두 통합됨

## [P3] 디바이스-모델 호환성 프로파일 설계
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 향후 다중 모델/디바이스 확장 시 자동 설정 결정을 위한 프로파일 시스템 설계
- **Acceptance Criteria**: 프로파일 스키마, 메모리 산출 공식 문서
- **Notes**: 당장 구현 불필요, 디바이스/모델 조합이 늘어날 때 착수
