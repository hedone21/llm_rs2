# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `src/backend/`, `src/models/`, `src/layers/`, `src/memory/`, `src/buffer/`, `src/bin/`, `src/resilience/`

---

## [P0] T1 Foundation 유닛 테스트 구현
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: T1 tier 기초 유닛 테스트 구현. 대상 모듈: Shape (생성, broadcast, 비교), Tensor (메타데이터, 슬라이싱), Buffer/DType (타입 변환, 크기 계산), Quant (Q4_0/Q8_0 양자화/역양자화), SharedBuffer (생성, 매핑), Galloc (할당/해제, 정렬)
- **Acceptance Criteria**: 각 모듈당 최소 3개 테스트 케이스, `cargo test` 전체 통과
- **Notes**: 테스터가 결과 검증 예정

## [P0] T2 Algorithm 유닛 테스트 구현
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: T1 테스트 완료
- **Description**: T2 tier 알고리즘 유닛 테스트 구현. 대상 모듈: KVCache (삽입, 조회, 크기 관리), EvictionPolicy (SlidingWindow, NoEviction), CacheManager (정책 적용, 메모리 한도), Attention (score 계산, masking), SystemMonitor (메트릭 수집)
- **Acceptance Criteria**: 각 모듈당 최소 3개 테스트 케이스, `cargo test` 전체 통과
- **Notes**: 테스터가 결과 검증 예정

## [P1] Resilience Manager generate.rs 통합 구현
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 아키텍트의 통합 설계 완료
- **Description**: 아키텍트의 설계에 따라 Resilience Manager를 generate.rs 메인 루프에 통합. D-Bus 신호 수신 스레드, 의사결정 로직, 백엔드 전환/배치 조정/캐시 정리 동작 구현
- **Acceptance Criteria**: generate 실행 시 resilience 신호에 반응, graceful degradation 동작, 유닛 테스트 포함
- **Notes**: tokio 사용 금지 — std::thread + std::sync 사용

## [P2] 컴파일러 경고 수정
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: 현재 컴파일 시 발생하는 경고 수정. 대상: x86.rs (unused imports), kv_cache.rs (dead code), tensor.rs (unused variables), sys_monitor.rs (unused fields), buffer.rs (unused methods) 등
- **Acceptance Criteria**: `cargo check` 경고 0건
- **Notes**: backlog에서 이관됨
