# Frontend Developer TODO

> **역할**: 웹 대시보드 개발, 벤치마크 시각화, 모니터링 UI
> **소유 영역**: `dashboard/`, `scripts/visualize_*.py`, `scripts/plot_*.py`

---

## [P1] Dashboard Experiments 탭 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Round 2 실험 완료 (결과 JSONL 축적 후)
- **Description**: 기존 Flask 대시보드에 Experiments 탭 추가. JSONL 결과를 파싱하여 인터랙티브 비교 제공
- **Acceptance Criteria**:
  - `dashboard/backend/experiment_parser.py`: JSONL 파싱 (per-token + summary)
  - `/api/experiments` 엔드포인트: 실험 목록, 개별 결과, 비교 데이터
  - Experiments 탭 UI:
    - 실험 목록 테이블 (ID, Signal, Eviction, Tokens, EMR, TBT%, RSS)
    - 개별 실험 상세: TBT 시계열 (Plotly), RSS 시계열, 품질 메트릭 카드
    - 비교 모드: baseline vs experiment 겹쳐 보기 (기존 Compare 탭 패턴 활용)
    - 신호 주입 시점 수직선 + eviction 이벤트 마커
  - 기존 대시보드 탭 (Overview, Table, Detail, ...) 영향 없음
- **Notes**: CLI 스크립트 (compare.py, plot_*.py)가 먼저 완성된 후 대시보드로 확장. Plotly.js로 인터랙티브 줌/필터 제공

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

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

## [P0] Experiment 분석 스크립트 구현
- **Status**: DONE
- **Notes**: quality_metrics.py (FDT, EMR, Suffix EMR, ROUGE-L, BLEU-4, Top-K Overlap), compare.py, round_report.py 구현 완료. 외부 의존성 없이 ROUGE-L/BLEU-4 직접 구현.

## [P0] Experiment 시각화 스크립트 구현
- **Status**: DONE
- **Notes**: plot_tbt_timeline.py (baseline 밴드 + 실험 라인 + 신호/eviction 마커), plot_rss_timeline.py 구현 완료. matplotlib 기반.

</details>
