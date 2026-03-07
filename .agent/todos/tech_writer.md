# Technical Writer TODO

> **역할**: 기술 문서 작성, API 가이드, 설계 문서 리뷰, 기술 조사
> **소유 영역**: `docs/` (00~26), `README.md`, `PROJECT_CONTEXT.md`, `results/GUIDE.md`

---

## [P0] Experiment 프레임워크 사용 가이드
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Rust Dev의 Experiment Mode, Frontend Dev의 분석 스크립트 완료
- **Description**: Resilience 실험 프레임워크 사용법 문서화. `docs/28_experiment_guide.md`
- **Acceptance Criteria**:
  - 실험 목적과 설계 원칙 설명
  - CLI 플래그 레퍼런스 (--experiment-schedule, --experiment-output, --experiment-sample-interval, --greedy)
  - 스케줄 JSON 포맷 명세 + 작성 가이드
  - JSONL 출력 스키마 명세 (per-token + summary)
  - 분석 스크립트 사용법 (compare.py, round_report.py, plot_*.py)
  - 품질 메트릭 설명 (FDT, EMR, ROUGE-L, BLEU-4, Top-K Overlap)
  - end-to-end 예제 워크플로우
- **Notes**: `experiments/PLAN.md`를 기반으로 사용자 가이드 형태로 재구성

## [P0] Experiment FINDINGS 기록
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 각 Round 실험 완료 시마다
- **Description**: 각 Round 실험 완료 후 `experiments/FINDINGS.md`에 가설/결과/인사이트/다음 방향 기록
- **Acceptance Criteria**:
  - Round별 섹션 작성 (가설, 결과 수치, 인사이트, 다음 실험 방향)
  - 핵심 수치 데이터 포함 (EMR, FDT, TBT%, RSS delta 등)
  - 이전 Round와의 연결성 기술
- **Notes**: 실험 실행과 병행하여 지속적으로 업데이트

---

## [P2] IPC Transport 문서화
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음 (IPC Transport 구현 완료)
- **Description**: SignalTransport 추상화 계층 문서화. D-Bus(Linux) vs UnixSocket(Android) 선택 로직, 메시지 포맷, 설정 방법, 트러블슈팅
- **Acceptance Criteria**: 플랫폼별 설정 가이드, 메시지 포맷 명세, 문제 해결 가이드
- **Notes**: docs/20_dbus_ipc_spec.md 업데이트 또는 별도 문서

## [P2] Manager 서비스 운영 가이드
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Manager 서비스 구현 완료
- **Description**: Manager 서비스 빌드, 설치, 설정(TOML), systemd 등록, 모니터링 방법 문서화
- **Acceptance Criteria**: 운영 가이드 완성, 새 사용자가 문서만으로 Manager 배포 가능
- **Notes**: Part 2 완성 시점에 작성

## [P2] 디바이스 포팅 가이드
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Device Registry 시스템, 범용 배포 스크립트
- **Description**: 새로운 디바이스를 추가하는 방법 문서화
- **Acceptance Criteria**: 포팅 체크리스트, 디바이스 추가 step-by-step 가이드
- **Notes**: Part 3 완성 시점에 작성

## [P3] 다중 모델 지원 가이드 작성
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 다중 모델 실제 테스트 시점
- **Description**: 다양한 모델 크기(1B/3B/7B/8B)의 다운로드, 양자화, 배포, 실행 방법 문서화
- **Acceptance Criteria**: 모델별 설정 가이드, 디바이스별 호환성 표
- **Notes**: 실제 테스트 데이터 확보 후 작성
