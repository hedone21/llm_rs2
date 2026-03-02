# Tester TODO

> **역할**: 테스트 전략 수립, 테스트 케이스 설계, QA 검증, 품질 게이트 관리
> **소유 영역**: `tests/`, `docs/14_component_status.md`, `scripts/update_test_status.py`, `.agent/rules/TESTING_STRATEGY.md`

---

## [P0] T1/T2 테스트 결과 검증 및 품질 게이트 업데이트
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: Rust 개발자의 T1/T2 테스트 구현 완료
- **Description**: Rust 개발자가 구현한 T1 Foundation, T2 Algorithm 유닛 테스트 결과를 검증. 테스트 커버리지 확인, 엣지 케이스 누락 점검, 품질 게이트 기준 충족 여부 판단. `docs/14_component_status.md` 업데이트
- **Acceptance Criteria**: 테스트 결과 리포트 작성, 품질 게이트 업데이트, 누락 테스트 케이스 목록 (있을 경우)
- **Notes**: Rust 개발자 완료 후 즉시 착수

## [P1] Resilience 모듈 통합 테스트 설계
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 아키텍트의 Resilience 통합 설계
- **Description**: Resilience Manager 통합 후 통합 테스트 설계. 시나리오: 정상 동작, 온도 경고 시 반응, 메모리 부족 시 반응, D-Bus 연결 실패 시 fallback, 다중 신호 동시 수신
- **Acceptance Criteria**: 테스트 시나리오 문서, 각 시나리오별 입력/예상출력/검증방법 명세
- **Notes**: 통합 설계 완료 후 착수, 디바이스 테스트와 호스트 테스트 구분

## [P1] 디바이스 스트레스 테스트 (Resilience 신호 포함)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Resilience Manager 통합 구현 완료
- **Description**: 실제 Android 디바이스에서 장시간 추론 실행하며 resilience 신호 동작 검증. 온도 상승에 따른 자동 throttling, 메모리 압박 시 캐시 정리, 복구 후 정상 동작 확인
- **Acceptance Criteria**: 1시간 연속 추론 안정 동작, 메모리 누수 없음, 비정상 종료 0건
- **Notes**: `stress_test_adb.py` 확장 필요

## [P2] CI 자동화 훅 설정
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: T1/T2 테스트 안정화
- **Description**: cargo test를 CI 파이프라인에 통합. pre-commit 또는 PR 단위로 T1 테스트 자동 실행, 결과 리포트 자동 생성
- **Acceptance Criteria**: CI에서 자동 테스트 실행, 실패 시 알림, 결과 리포트 아카이브
- **Notes**: GitHub Actions 또는 로컬 훅 중 선택 필요
