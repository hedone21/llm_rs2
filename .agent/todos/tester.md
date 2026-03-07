# Tester TODO

> **역할**: 테스트 전략 수립, 테스트 케이스 설계, QA 검증, 품질 게이트 관리
> **소유 영역**: `engine/tests/`, `docs/14_component_status.md`, `scripts/update_test_status.py`, `.agent/rules/TESTING_STRATEGY.md`

---

## [P0] Experiment Baseline 실행 + JSONL 검증
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Rust Dev의 Experiment Mode 구현 완료
- **Description**: Baseline 실험 실행 및 JSONL 출력 포맷 검증
- **Acceptance Criteria**:
  - B-128 (128 토큰, none), B-512 (512 토큰, none), B-1024 (1024 토큰, none) 실행
  - B-512-sliding, B-512-h2o 실행 (eviction 오버헤드 baseline)
  - JSONL 출력 구조 검증: 모든 필드 존재, 타입 정확, summary 레코드 포함
  - sys 메트릭 정상 수집 확인 (rss_mb, cpu_pct, cpu_mhz, thermal_mc)
  - greedy 모드에서 동일 prompt → 동일 출력 재현성 확인 (2회 반복)
  - `experiments/results/` 에 결과 저장
- **Notes**: `experiments/PLAN.md` Section 5 Round 1 참조

## [P0] Round 2 단일 신호 실험 실행 + 분석
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Baseline 실행 완료, 분석 스크립트 완료 (Frontend Dev)
- **Description**: Round 2 전체 14 실험 실행. 속도(128tok) 5건 + 품질(512tok) 4건 + 메모리(1024tok) 5건
- **Acceptance Criteria**:
  - 14 실험 모두 정상 실행, JSONL 생성
  - compare.py로 각 실험 baseline 대비 보고서 생성
  - round_report.py로 Round 2 요약 테이블 생성
  - FINDINGS.md에 가설/결과/인사이트 기록
- **Notes**: `experiments/PLAN.md` Section 5 Round 2 참조. 예상 실행 시간 ~15분

## [P0] Round 3~5 실험 반복 실행
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Round 2 분석 완료
- **Description**: Round 2 인사이트 기반으로 Round 3(위치변수), Round 4(H2O sweep), Round 5(복합) 순차 실행
- **Acceptance Criteria**:
  - 각 Round별 실험 실행 + 보고서 + FINDINGS 업데이트
  - Round 간 인사이트 연결 (이전 Round 결과가 다음 Round 설계에 반영)
- **Notes**: Round 3~5는 Round 2 결과에 따라 조정 가능

---

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

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

## [P1] 디바이스 스트레스 테스트 (Resilience 신호 포함)
- **Status**: DONE
- **Notes**: Phase 6 (Resilience) 추가. signal_injector + 4개 스케줄. L2 통합 테스트 14개 통과

</details>
