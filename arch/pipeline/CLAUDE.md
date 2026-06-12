# arch/pipeline/ — 작업 가이드 (LLM용)

이 폴더는 `arch/pipeline_stage_design_v2.md`(이하 v2)를 처음 읽는 사람과 LLM이 이해하도록 돕는 **동반 문서 세트**다. v2가 정식 출처(SSOT)이고, 이 폴더의 문서는 비규범적 해설이다.

## 이 폴더를 편집할 때

- **v2를 복사하지 않는다.** 정확한 시그니처·불변식·enum variant는 v2에 링크한다. 여기서는 단순화 예시만 쓰고 `> 📘` 라벨을 붙인다.
- **이력 코드를 본문에 쓰지 않는다.** `G3`·`R1`·`α-K` 같은 grill 결정 ID는 이 폴더 문서에 노출하지 않는다. 독자가 v2에서 마주칠 때 "건너뛰어도 된다"는 안내만 `README.md`에 둔다.
- **도메인 용어는 `/CONTEXT.md`로 링크한다.** Format·Stage·Layer 등은 CONTEXT.md가 SSOT다. 여기서 재정의하지 않는다.
- v2가 바뀌면 영향받는 문서의 동기화 기준 날짜를 갱신하고 내용을 재검토한다.

## 문서 세트 맵

| 파일 | 유형 (Diátaxis) | 내용 | v2 매핑 |
|---|---|---|---|
| `README.md` | 인덱스 | 라우팅 + v2 읽기 안내 | §0.4 |
| `01_overview.md` | Explanation | 미션 + 3 지배원칙 + 레이어 구조 + front-door | §0~§2 |
| `02_backend_capability.md` | Explanation | Backend, capability, Hardware resolver | §3 |
| `03_format_vs_stage.md` | Explanation | Format(표현) ⊥ Stage(동작) 직교 | §4 |
| `04_pipeline_stage.md` | Explanation | PipelineStage, LifecyclePhase, 순서 책임 | §5.1~5.3 |
| `05_resilience.md` | Explanation | Pressure scalar ∥ Command discrete | §5.4 |
| `10_extending.md` | How-to | 새 backend / 새 KV Format 추가 실습 | §0.4~0.5 |
| `90_glossary.md` | Glossary | 설계 jargon + CONTEXT.md 링크 | — |

읽는 순서: `01 → 02 → 03 → 04 → 05`. `10`은 개념을 익힌 뒤 실습. `90`은 아무 때나 참조.

## 문장 기준 (이 폴더의 모든 산문에 적용)

**기술 문서 문장 원칙**
- 한 문단은 하나의 생각만 담는다. 문장은 평균 15단어, 최대 25단어.
- 능동태로 쓴다. 주어가 동작을 수행한다.
- 구체적이고 측정 가능한 표현을 쓴다. "크다", "좋다", "강력한" 같은 모호어를 피한다.
- 전문용어는 첫 등장에서 한 문장으로 설명하거나 `90_glossary.md`로 링크한다.
- 불필요한 단어를 뺀다. 주어와 동사를 가까이 둔다.

**AI 말투 회피**
- em-dash(—)를 남용하지 않는다. 마침표·쉼표·콜론·괄호로 대체한다.
- 삼단 나열(rule of three)을 리듬 목적으로 반복하지 않는다.
- 극적 단언을 쓰지 않는다. "그게 전부다", "답은 ~에 있다", "바로 ~이다", "핵심은", "결국", "사실상" 같은 표현 대신 사실과 근거로 서술한다.
- 같은 문장 시작이나 같은 리듬을 반복하지 않는다. 문장 도입을 다양화한다.
- 헤더·불릿·번호를 과용하지 않는다. 산문으로 충분하면 산문으로 쓴다.

## 문서 형식 규약

- 모든 문서는 본문 최상단(배너 다음)에 `## 목차`를 둔다.
- 각 문서 배너에 유형·SSOT·비규범성·동기화 기준 날짜를 명시한다.
- 문서 간 이동은 순서대로 읽도록 안내한다. 문서 안에서 "뒤 섹션부터 읽으라"고 지시하지 않는다.
