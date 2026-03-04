# TODO Management System

Antigravity 프로젝트의 팀 역할 정의 및 업무 추적 시스템.

---

## 역할 정의

### 1. 아키텍트 (Architect)

- **책임**: 시스템 설계, 트레이트/인터페이스 정의, 모듈 간 의존성 관리, 기술 결정
- **소유 영역**: `engine/src/core/`, `shared/`, Cargo workspace 구조, `ARCHITECTURE.md`
- **도구**: `cargo check`, 설계 문서, 다이어그램
- **산출물**: 설계 문서, RFC, 트레이트 정의, 아키텍처 다이어그램
- **자율 결정**: 내부 트레이트 리팩터링, 모듈 구조 변경
- **PM 승인 필요**: 새로운 외부 의존성 추가, 주요 API 변경, 새 모듈 생성
- **협업**: Rust 개발자에게 구현 가이드 제공, 테크니컬 라이터와 문서 동기화

### 2. Rust 개발자 (Rust Developer / LLM Developer)

- **책임**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
- **소유 영역**: `engine/src/`, `manager/`, `shared/`
- **도구**: `cargo build/test/check`, `sanity_check.sh`, 프로파일링 스크립트
- **산출물**: 기능 구현 코드, 유닛 테스트, 벤치마크 결과
- **자율 결정**: 함수 내부 최적화, 버그 수정, 유닛 테스트 작성
- **PM 승인 필요**: 새 바이너리 추가, Backend 트레이트 변경 (아키텍트와 협의), `.cl` 커널 수정
- **협업**: 아키텍트의 설계에 따라 구현, 테스터에게 테스트 포인트 전달

### 3. 프론트엔드 개발자 (Frontend Developer / Dashboard Developer)

- **책임**: 웹 대시보드 개발, 벤치마크 시각화, 모니터링 UI
- **소유 영역**: `dashboard/`, `scripts/visualize_*.py`, `scripts/plot_*.py`
- **도구**: Flask, Plotly.js, Python, HTML/CSS/JS
- **산출물**: 대시보드 UI, 차트/그래프, REST API 엔드포인트
- **자율 결정**: UI 레이아웃, 차트 스타일, 프론트엔드 라이브러리 선택
- **PM 승인 필요**: 새 REST 엔드포인트 스키마 변경, 외부 JS 의존성 추가
- **협업**: Rust 개발자로부터 JSON 스키마 수신, 테스터와 시각화 검증

### 4. 테스터 (Tester)

- **책임**: 테스트 전략 수립, 테스트 케이스 설계, QA 검증, 품질 게이트 관리
- **소유 영역**: `engine/tests/`, `docs/14_component_status.md`, `scripts/update_test_status.py`, `.agent/rules/TESTING_STRATEGY.md`
- **도구**: `cargo test`, `run_android.sh`, `stress_test_adb.py`, `test_backend`
- **산출물**: 테스트 케이스, 테스트 결과 리포트, 품질 게이트 업데이트
- **자율 결정**: 테스트 케이스 추가, 테스트 스크립트 수정, 품질 게이트 기준 조정
- **PM 승인 필요**: 테스트 티어 변경, 게이트 기준 완화
- **협업**: Rust 개발자에게 테스트 요구사항 전달, 결과를 PM에 보고

### 5. 테크니컬 라이터 / 연구자 (Technical Writer / Researcher)

- **책임**: 기술 문서 작성, API 가이드, 설계 문서 리뷰, 기술 조사
- **소유 영역**: `docs/` (00~22), `README.md`, `PROJECT_CONTEXT.md`, `results/GUIDE.md`
- **도구**: Markdown, 다이어그램 도구, 웹 리서치
- **산출물**: 설계 문서, 사용자 가이드, API 레퍼런스, 기술 조사 보고서
- **자율 결정**: 문서 구조, 용어 표준화, 기존 문서 업데이트
- **PM 승인 필요**: 새 문서 번호 할당, 문서 삭제, 외부 공개 문서
- **협업**: 아키텍트/Rust 개발자로부터 기술 내용 확인, 모든 역할에 문서 제공

---

## TODO 형식

각 역할별 TODO 파일에서 사용하는 공통 형식:

```markdown
## [P0] 작업 제목
- **Status**: TODO | IN_PROGRESS | BLOCKED | DONE
- **Sprint**: current | next | backlog
- **Dependencies**: (선행 작업이나 다른 역할)
- **Description**: 구체적인 작업 내용
- **Acceptance Criteria**: 완료 조건
- **Notes**: (추가 메모)
```

### 우선순위

| 레벨 | 의미 | 대응 시간 |
|------|------|-----------|
| P0 | 긴급/블로커 | 즉시 |
| P1 | 높음 | 현재 스프린트 |
| P2 | 보통 | 다음 스프린트 |
| P3 | 낮음 | 백로그 |

### 상태 흐름

```
TODO → IN_PROGRESS → DONE
         ↓
      BLOCKED → IN_PROGRESS → DONE
```

---

## 워크플로우 규칙

1. **작업 배정**: PM이 backlog에서 역할별 TODO로 이동
2. **작업 시작**: 담당자가 Status를 `IN_PROGRESS`로 변경
3. **블로커 발생**: Status를 `BLOCKED`로 변경, Dependencies에 블로커 명시
4. **작업 완료**: Status를 `DONE`로 변경, Acceptance Criteria 충족 확인
5. **리뷰**: PM이 DONE 항목 확인 후 아카이브

### 파일 구조

| 파일 | 용도 |
|------|------|
| `backlog.md` | 미배정 작업 백로그 |
| `architect.md` | 아키텍트 TODO |
| `rust_developer.md` | Rust 개발자 TODO |
| `frontend_developer.md` | 프론트엔드 개발자 TODO |
| `tester.md` | 테스터 TODO |
| `tech_writer.md` | 테크니컬 라이터 TODO |
