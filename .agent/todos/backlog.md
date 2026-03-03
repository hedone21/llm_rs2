# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [P2] 컴파일러 경고 정리
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: unused imports, dead code 등 컴파일러 경고 수정. 대상 파일: x86.rs, kv_cache.rs, tensor.rs, sys_monitor.rs, buffer.rs 등 약 6개 파일
- **Acceptance Criteria**: `cargo check 2>&1 | grep warning` 결과 0건
- **Notes**: Rust 개발자에게 배정 예정

## [P2] 미커밋 파일 정리
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: eviction_memory_test 관련 미커밋 파일 3개 처리 (results/data/eviction_memory_test.json, scripts/plot_eviction_memory.py, tests/test_eviction_memory.rs). 커밋 또는 .gitignore 추가 결정 필요
- **Acceptance Criteria**: `git status`에서 untracked 파일 0건
- **Notes**: PM 결정 필요 — 커밋 vs gitignore

## [P1] SnapKV 완전 구현
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 아키텍트의 attention score 노출 인터페이스 설계 (DONE)
- **Description**: 현재 stub 상태인 SnapKV를 실제 attention score 기반으로 완전 구현. 중요도 낮은 KV 엔트리를 선택적으로 제거하는 지능형 캐시 관리
- **Acceptance Criteria**: attention score 기반 KV 엔트리 선택/제거 동작, 유닛 테스트 통과
- **Notes**: AttentionScoreAccumulator + evict_with_scores 구현 완료. 인터페이스 설계 + 구현 모두 완료됨

## [P2] GPU 전용 버퍼 prune_prefix 지원
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 아키텍트의 GPU 버퍼 전략 설계
- **Description**: OpenCL 전용 버퍼에서 prune_prefix 연산 지원. 현재 CPU 매핑 가능 버퍼만 지원
- **Acceptance Criteria**: GPU 전용 버퍼에서 prune_prefix 정상 동작, 메모리 누수 없음
- **Notes**: 아키텍트 설계 완료 후 Rust 개발자에게 배정
