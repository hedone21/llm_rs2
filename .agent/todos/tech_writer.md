# Technical Writer TODO

> **역할**: 기술 문서 작성, API 가이드, 설계 문서 리뷰, 기술 조사
> **소유 영역**: `docs/` (00~22), `README.md`, `PROJECT_CONTEXT.md`, `results/GUIDE.md`

---

## [P1] Resilience Phase 1 통합 가이드 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 아키텍트의 Resilience 통합 설계 완료
- **Description**: Resilience 시스템의 Phase 1 통합 내용을 문서화. D-Bus 리스너 설정, 시스템 모니터 구성, generate.rs 통합 방법, 설정 옵션, 트러블슈팅 가이드 포함
- **Acceptance Criteria**: 문서 완성, 아키텍트/Rust 개발자 검토 통과, 새 사용자가 문서만으로 설정 가능
- **Notes**: `docs/` 시리즈에 추가 (번호 할당은 PM 승인 필요)

## [P1] ARCHITECTURE.md에 resilience 흐름 추가
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 아키텍트의 Resilience 통합 설계 완료
- **Description**: ARCHITECTURE.md에 resilience 모듈의 위치, 데이터 흐름, 다른 모듈과의 상호작용을 추가. 기존 아키텍처 다이어그램 업데이트
- **Acceptance Criteria**: ARCHITECTURE.md 업데이트, 아키텍트 검토 통과
- **Notes**: 아키텍트와 긴밀히 협업

## [P2] 트러블슈팅 가이드 작성
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: T1/T2 테스트 완료, Resilience 통합 완료
- **Description**: 빌드 오류, 런타임 오류, 디바이스 이슈 등 자주 발생하는 문제와 해결 방법 문서화. OpenCL 초기화 실패, 메모리 부족, D-Bus 연결 오류 등 포함
- **Acceptance Criteria**: 최소 10개 이상의 문제/해결책 쌍, 검색 가능한 구조
- **Notes**: 테스터와 Rust 개발자로부터 이슈 수집

## [P2] API 레퍼런스 업데이트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Resilience 통합 완료
- **Description**: Backend trait 변경사항, 새 resilience API, 설정 옵션 등 API 레퍼런스 업데이트. `docs/02_core_abstractions.md` 등 관련 문서 갱신
- **Acceptance Criteria**: API 변경 사항 100% 반영, 코드 예제 포함
- **Notes**: `cargo doc`으로 생성되는 API 문서와 일관성 유지
