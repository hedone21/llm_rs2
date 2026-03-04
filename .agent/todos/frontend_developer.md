# Frontend Developer TODO

> **역할**: 웹 대시보드 개발, 벤치마크 시각화, 모니터링 UI
> **소유 영역**: `dashboard/`, `scripts/visualize_*.py`, `scripts/plot_*.py`

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
