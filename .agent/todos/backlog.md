# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [P3] 다중 모델 사이즈 검증 테스트 매트릭스
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 다중 디바이스 포팅 완료
- **Description**: Llama 3.2 1B/3B 및 향후 7B/8B 모델에 대한 디바이스별 테스트 매트릭스 정의
- **Acceptance Criteria**: 매트릭스 문서, 디바이스별 최대 지원 모델 크기 명시
- **Notes**: 실제 테스트는 디바이스 확보 후 진행

## [P3] ThermalCollector zone 패턴 매칭 auto-discovery
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 현재 `zone_types`는 exact match. substring/keyword 매칭으로 확장하면 다양한 장치 커버 가능
- **Acceptance Criteria**: contains 기반 패턴 매칭, 기존 exact match와 공존, 테스트 추가
- **Notes**: 필요성 미확정. 실제 다중 장치 배포 시점에 재평가

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

- [P1] IPC Transport 추상화 설계 — 커밋 c2b7c64
- [P2] Manager 서비스 프로젝트 스캐폴딩 — 커밋 95af0a3
- [P2] Device Registry 시스템 — devices.toml + run_device.py

</details>
