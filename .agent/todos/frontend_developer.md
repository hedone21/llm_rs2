# Frontend Developer TODO

> **역할**: 웹 대시보드 개발, 벤치마크 시각화, 모니터링 UI
> **소유 영역**: `dashboard/`, `scripts/visualize_*.py`, `scripts/plot_*.py`

---

## [P0] Experiment 분석 스크립트 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Rust Dev의 Experiment Mode 구현 완료 (JSONL 스키마 확정)
- **Description**: 실험 결과 JSONL을 분석하여 속도/품질/리소스 메트릭을 계산하고 보고서를 생성하는 Python 스크립트 세트
- **Acceptance Criteria**:
  - `experiments/analysis/quality_metrics.py`: FDT, EMR, Suffix EMR, ROUGE-L, BLEU-4, Top-K Overlap 계산
  - `experiments/analysis/compare.py`: baseline vs experiment 비교, 속도/품질/리소스 메트릭 출력, Markdown 보고서 생성
  - `experiments/analysis/round_report.py`: Round 전체 요약 테이블 (ID, Signal, Evict, TBT%, EMR, FDT, ROUGE-L, RSS, ...)
  - `experiments/analysis/requirements.txt`: rouge-score, nltk, matplotlib
  - JSONL 파싱: per-token 레코드 + _summary 레코드 처리
- **Notes**: `experiments/PLAN.md` Section 6 참조. quality_metrics.py가 핵심 라이브러리

## [P0] Experiment 시각화 스크립트 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: compare.py 구현 완료
- **Description**: 실험 결과를 시각화하는 matplotlib 기반 그래프 생성 스크립트
- **Acceptance Criteria**:
  - `experiments/analysis/plot_tbt_timeline.py`: X=토큰위치, Y=TBT(ms). baseline 밴드 + 실험 라인 + 신호 주입 수직선 + eviction 마커
  - `experiments/analysis/plot_rss_timeline.py`: X=토큰위치, Y=RSS(MB). eviction 전후 drop 관측
  - 출력: `experiments/reports/plots/` PNG 파일
  - 다중 실험 겹쳐 그리기 지원 (`--experiments exp1.jsonl exp2.jsonl`)
- **Notes**: `experiments/PLAN.md` Section 6.3 참조

---

## [P2] Thermal throttling 감지 및 알림 UI
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Resilience 신호 시각화 완료
- **Description**: 디바이스 온도가 임계값 초과 시 대시보드에서 알림 표시. 색상 변경 (초록→노랑→빨강), 알림 배너, 자동 새로고침
- **Acceptance Criteria**: 온도 임계값 초과 시 시각적 알림, 3단계 색상 코딩, 알림 히스토리
- **Notes**: 임계값은 Rust 개발자/아키텍트와 협의

## [P2] 다중 디바이스 비교 대시보드
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Device Registry 시스템, 다중 모델 벤치마크 데이터
- **Description**: 디바이스 × 모델 크기 × 양자화 조합별 성능 비교 차트
- **Acceptance Criteria**: 디바이스 드롭다운 선택, 비교 차트 렌더링, 호환성 매트릭스 표시
- **Notes**: results/data/ JSON 기반, 새 비교 스키마 필요

## [P3] 실시간 추론 스트리밍 모드
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 대시보드에서 추론 결과를 실시간 스트리밍으로 표시. WebSocket 또는 SSE 기반
- **Acceptance Criteria**: 실시간 토큰 출력 표시, 지연 < 100ms, 성능 메트릭 동시 갱신
- **Notes**: 우선순위 낮음, 백로그 유지
