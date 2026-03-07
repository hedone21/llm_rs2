# Tester TODO

> **역할**: 테스트 전략 수립, 테스트 케이스 설계, QA 검증, 품질 게이트 관리
> **소유 영역**: `engine/tests/`, `docs/14_component_status.md`, `scripts/update_test_status.py`, `.agent/rules/TESTING_STRATEGY.md`

---

## [P2] 범용 디바이스 배포/테스트 스크립트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Device Registry 시스템
- **Description**: adb(Android)/ssh(Linux) 추상화된 배포 스크립트
- **Acceptance Criteria**: 단일 명령으로 다중 디바이스 테스트 실행, 결과 JSON 수집
- **Notes**: 기존 run_android.sh를 범용화

## [P2] Manager 서비스 통합 테스트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Manager 서비스 구현
- **Description**: Manager + LLM E2E 연동 테스트
- **Acceptance Criteria**: 각 시나리오 통과, 플랫폼별 검증
- **Notes**: MockTransport 기반 호스트 테스트 + 실 디바이스 테스트 분리

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

## [P0] Experiment Baseline 실행 + JSONL 검증
- **Status**: DONE
- **Notes**: 커밋 1a6633f. B-128/B-512/B-1024/B-512-sliding/B-512-h2o 실행, 20개 JSONL 검증 통과, 재현성 확인

## [P0] Round 2 단일 신호 실험 실행 + 분석
- **Status**: DONE
- **Notes**: 커밋 1a6633f. 14 실험 완료. 속도(T-C-32: +174%), 품질(H2O EMR=0.593), 메모리(R-C-768: EMR=1.0) 확인

## [P0] Round 3~5 실험 반복 실행
- **Status**: DONE
- **Notes**: Round 3(8실험), Round 4(9실험), Round 5(9실험+B-2048) 완료. 총 45개 실험. 핵심 발견: eviction 위치가 품질 최대 결정요인, H2O recent_window만 유의미한 파라미터, 복합 제약은 속도/품질 독립 영향

## [P1] 디바이스 스트레스 테스트 (Resilience 신호 포함)
- **Status**: DONE
- **Notes**: Phase 6 (Resilience) 추가. signal_injector + 4개 스케줄. L2 통합 테스트 14개 통과

</details>
