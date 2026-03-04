# Tester TODO

> **역할**: 테스트 전략 수립, 테스트 케이스 설계, QA 검증, 품질 게이트 관리
> **소유 영역**: `engine/tests/`, `docs/14_component_status.md`, `scripts/update_test_status.py`, `.agent/rules/TESTING_STRATEGY.md`

---

## [P1] 디바이스 스트레스 테스트 (Resilience 신호 포함)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Resilience Manager 통합 구현 완료
- **Description**: 실제 Android 디바이스에서 장시간 추론 실행하며 resilience 신호 동작 검증. 온도 상승에 따른 자동 throttling, 메모리 압박 시 캐시 정리, 복구 후 정상 동작 확인
- **Acceptance Criteria**: 1시간 연속 추론 안정 동작, 메모리 누수 없음, 비정상 종료 0건
- **Notes**: `scripts/stress_test_device.py` 확장 필요

## [P2] 범용 디바이스 배포/테스트 스크립트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Device Registry 시스템
- **Description**: adb(Android)/ssh(Linux) 추상화된 배포 스크립트. devices.toml에서 디바이스 목록 읽어 자동 빌드→배포→테스트→결과수집 파이프라인
- **Acceptance Criteria**: 단일 명령으로 다중 디바이스 테스트 실행, 결과 JSON 수집
- **Notes**: 기존 run_android.sh를 범용화

## [P2] Manager 서비스 통합 테스트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Manager 서비스 구현
- **Description**: Manager + LLM E2E 연동 테스트. 시나리오: 정상 동작, 메모리 부족 시그널→eviction, 온도 경고→throttle
- **Acceptance Criteria**: 각 시나리오 통과, 플랫폼별(Linux D-Bus, Android UnixSocket) 검증
- **Notes**: MockTransport 기반 호스트 테스트 + 실 디바이스 테스트 분리
