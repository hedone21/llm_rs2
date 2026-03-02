# Frontend Developer TODO

> **역할**: 웹 대시보드 개발, 벤치마크 시각화, 모니터링 UI
> **소유 영역**: `web_dashboard/`, `scripts/visualize_*.py`, `scripts/plot_*.py`

---

## [P1] 컴포넌트 게이트 시각화 UI (Gates 탭)
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: update_test_status.py JSON 출력
- **Description**: 대시보드에 Gates 탭 추가. 컴포넌트별 Quality Gate 상태 테이블, 요약 카드, Pass Rate 히스토리 차트 표시. API 엔드포인트 /api/gates, gate_parser.py, gates.js 구현.
- **Acceptance Criteria**: Gates 탭에서 컴포넌트 상태, 요약 카드, 히스토리 차트 표시
- **Notes**: results/data/component_gates.json → /api/gates → Gates.load()

## [P1] Resilience 신호 시각화 UI 추가
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: Rust 개발자의 Resilience Manager 통합 완료 (JSON 출력 스키마 확정)
- **Description**: 대시보드에 resilience 신호 시각화 추가. 온도, 메모리 사용량, CPU 부하 등 시스템 메트릭을 실시간 차트로 표시. 백엔드 전환 이벤트, 배치 크기 조정 이벤트를 타임라인에 마킹
- **Acceptance Criteria**: 대시보드에서 resilience 메트릭 차트 표시, 이벤트 타임라인 동작, JSON 데이터 정상 파싱
- **Notes**: Rust 개발자로부터 JSON 스키마 수신 후 착수

## [P2] Thermal throttling 감지 및 알림 UI
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Resilience 신호 시각화 완료
- **Description**: 디바이스 온도가 임계값 초과 시 대시보드에서 알림 표시. 색상 변경 (초록→노랑→빨강), 알림 배너, 자동 새로고침
- **Acceptance Criteria**: 온도 임계값 초과 시 시각적 알림, 3단계 색상 코딩, 알림 히스토리
- **Notes**: 임계값은 Rust 개발자/아키텍트와 협의

## [P3] 실시간 추론 스트리밍 모드
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 대시보드에서 추론 결과를 실시간 스트리밍으로 표시. WebSocket 또는 SSE 기반, 토큰 단위 출력, 성능 메트릭 동시 표시
- **Acceptance Criteria**: 실시간 토큰 출력 표시, 지연 < 100ms, 성능 메트릭 동시 갱신
- **Notes**: 우선순위 낮음, 백로그 유지
