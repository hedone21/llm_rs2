# Architect TODO

> **역할**: 시스템 설계, 트레이트/인터페이스 정의, 모듈 간 의존성 관리, 기술 결정
> **소유 영역**: `engine/src/core/`, `shared/`, Cargo workspace 구조, `ARCHITECTURE.md`

---

## [P3] 디바이스-모델 호환성 프로파일 설계
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 향후 다중 모델/디바이스 확장 시 자동 설정 결정을 위한 프로파일 시스템 설계
- **Acceptance Criteria**: 프로파일 스키마, 메모리 산출 공식 문서
- **Notes**: 당장 구현 불필요, 디바이스/모델 조합이 늘어날 때 착수

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

## [P1] IPC Transport 추상화 설계
- **Status**: DONE
- **Notes**: 커밋 c2b7c64. SignalTransport trait + 3종 transport

## [P2] Cargo workspace 구조 설계 (Manager 서비스)
- **Status**: DONE
- **Notes**: 커밋 95af0a3. engine/shared/manager 3-crate workspace

## [P1] Hybrid 추론 + Eviction/Resilience 통합 설계
- **Status**: DONE
- **Notes**: CacheManager + Evict + SwitchBackend(양방향) + Throttle + Suspend 통합

</details>
