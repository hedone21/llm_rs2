# Architect TODO

> **역할**: 시스템 설계, 트레이트/인터페이스 정의, 모듈 간 의존성 관리, 기술 결정
> **소유 영역**: `src/core/`, `ARCHITECTURE.md`, 모듈 구조

---

## [P1] Resilience 시스템 generate.rs 통합 설계
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: Resilience Manager (D-Bus 기반 시스템 모니터링)를 generate.rs 메인 추론 루프에 통합하기 위한 설계. 신호 수신 → 의사결정 → 동작(백엔드 전환, 배치 축소, 캐시 정리) 흐름 정의. 스레드 모델, 에러 핸들링, fallback 전략 포함
- **Acceptance Criteria**: 통합 설계 문서 완성, 시퀀스 다이어그램, Rust 개발자가 구현 가능한 수준의 명세
- **Notes**: Rust 개발자의 구현과 테스터의 통합 테스트에 선행

## [P2] SnapKV attention score 노출 인터페이스 설계
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: 현재 attention 계산에서 score를 외부로 노출하는 인터페이스 설계. SnapKV가 attention score를 기반으로 중요 KV 엔트리를 판별하기 위해 필요. Backend trait 또는 별도 trait로 노출할지 결정
- **Acceptance Criteria**: 인터페이스 설계 문서, trait 시그니처 제안, 성능 영향 분석
- **Notes**: backlog의 "SnapKV 완전 구현"의 선행 작업

## [P2] GPU 버퍼 prune_prefix 전략 설계
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: OpenCL 전용 버퍼(CL_MEM_ALLOC_HOST_PTR 미사용)에서 prune_prefix를 지원하기 위한 전략 설계. GPU 커널 기반 데이터 이동 vs 재할당, 메모리 관리 방안
- **Acceptance Criteria**: 설계 문서, 선택한 접근법과 근거, 구현 가이드
- **Notes**: enqueue_copy_buffer + 오버랩 시 temp 버퍼 방식 채택. cl_mem() 버그 수정, GPU 테스트 5개 추가, docs/11 문서화 완료
