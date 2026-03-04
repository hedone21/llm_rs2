# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [P1] IPC Transport 추상화 설계
- **Status**: DONE
- **Sprint**: current
- **Description**: D-Bus 외에 Android용 Unix Domain Socket 전송 계층 지원. `SignalTransport` trait 정의
- **Notes**: 커밋 c2b7c64에서 구현 완료. 아키텍트/Rust 개발자 TODO에서도 DONE 처리됨

## [P2] Manager 서비스 프로젝트 스캐폴딩
- **Status**: DONE
- **Sprint**: next
- **Description**: Manager 서비스용 Rust 프로젝트 생성. signal 타입 공유 crate 분리, Cargo workspace 구성
- **Notes**: 커밋 95af0a3에서 구현 완료. `manager/`, `shared/` crate 생성, mock_manager 이전

## [P2] Device Registry 시스템
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: TOML 기반 디바이스 목록 관리 (이름, 연결방식, SoC, 메모리, GPU, 지원 기능). 테스트 스크립트에서 참조하여 자동 배포/테스트 수행
- **Acceptance Criteria**: devices.toml 스키마 정의, 파서 구현, 기존 run_android.sh와 연동
- **Notes**: Part 3 기초 작업

## [P3] 다중 모델 사이즈 검증 테스트 매트릭스
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 다중 디바이스 포팅 완료
- **Description**: Llama 3.2 1B/3B 및 향후 7B/8B 모델에 대한 디바이스별 테스트 매트릭스 정의
- **Acceptance Criteria**: 매트릭스 문서, 디바이스별 최대 지원 모델 크기 명시
- **Notes**: 실제 테스트는 디바이스 확보 후 진행
