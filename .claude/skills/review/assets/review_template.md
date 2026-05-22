# Review — `<Plan/Design/Decision 제목>`

**스코프**: Plan 사전 리뷰 | Design 리뷰 | Decision 재검토 | 빠른 sanity-check
**대상**: `<파일/모듈/sprint/spec ID>`

---

## 1. TL;DR

- 추천: `<옵션 / 방향>` (Status: Proposed | Accepted | Conditional | Superseded)
- 가장 큰 리스크: `<요약>` (RPN <NN>)
- 게이트: `<통과 조건 수치>`

## 2. Reviewed Items

- 읽은 파일·라인:
  - `<path>:<L>-<L>` — `<무엇을 봤는가>`
- 인용한 실측 데이터:
  - `<Round / Phase / Sprint>` — `<수치>`
- 검토했지만 기각한 후보:
  - `<후보명>` — `<1줄 기각 사유>`
- 본 프로젝트 invariant·spec 확인 결과:
  - `<INV-XXX / §13.8-X>` — `<영향 / 무영향>`

## 3. Design / Functionality / Tests / Backend Coverage

- **Design**: L1~L5 layering 일치 — `<O/X + 사유>`
- **Functionality**: prefill / decode 영향 — `<범위>`
- **Tests**: host cargo test + S25 device + Jetson CUDA — `<커버리지>`
- **Backend Coverage**: NEON / Adreno / CUDA — `<회귀 가능 영역>`

## 4. Tradeoffs (ATAM)

- **Sensitivity points** (작은 변경이 큰 영향):
  - `<예: DK=128 per-thread 32 float4 임계>`
- **Tradeoff points** (품질 희생):
  - `<예: TBT 개선 vs 메모리 spike>`

## 5. Risks

**정성 서술 (RPN < 100)**:
- [경미] `<설명>`
- [경미] `<설명>`

**정량 (RPN ≥ 100)**:
- [중간 RPN <NN>] `<설명>`
  - S <N> × O <N> × D <N>
  - 게이트: `<통과 조건>`
- [블로커 RPN ≥ 200] `<설명>` — **작업 차단**
  - S <N> × O <N> × D <N>
  - 게이트: `<premortem 필수 + 통과 조건>`

척도:
| 축 | 1점 | 5점 | 10점 |
|---|---|---|---|
| Severity | log 한 줄 | TBT +10% | PACT2026 차단 / crash |
| Occurrence | 검증 환경만 | 1B 모델 가끔 | 모든 backend 항상 |
| Detection | host test | S25 device test | logcat·production만 |

## 6. Alternatives Considered

| 후보 | 핵심 메커니즘 | Pros | Cons | 기각 사유 |
|------|---------------|------|------|-----------|
| (a) `<이름>` | `<요약>` | `<>` | `<>` | (선택) |
| (b) `<이름>` | `<요약>` | `<>` | `<>` | `<>` |
| (c) status quo (변경 없음) | — | `<>` | `<>` | `<>` |

## 7. Premortem

이 작업이 `<시점>`에 실패한다면 가장 가능성 높은 부고:

`<자유 서술 1단락 — 사후 분석 톤으로>`

## 8. Devil's Advocate Pass

- 내 결론이 틀렸다면 가장 가능성 높은 이유: `<>`.
- 이 결론이 맞으려면 `<>` 가정이 참이어야 한다. 이 가정의 약점: `<>`.

## 9. Recommendation

**선택**: `<단일안 + 핵심 한 줄>`
**Status**: Conditional — `<게이트>` 통과 시 Accepted
**근거**: `<3줄 이내>`
**관련 spec/arch**: `<INV-XXX / §13.5 Resolution Log V-?? / arch/<file>.md>`

## 10. Open Questions

- [Architect 답변 필요, `<phase 진입 전>`] `<질문>`
- [구현 중 해소 가능] `<질문>`
- [사용자 결정] `<질문>`

---

## 사용 팁

- **스코프별 부분집합**:
  - Plan 사전 리뷰: 1·2·5·6·7·9·10
  - Design 리뷰: 전 항목
  - Decision 재검토: 1·2·6·8·9
  - 빠른 sanity-check: 1·2·5·9
- **자기점검**: Reviewed Items / Alternatives / Premortem / DA Pass 4개는 강제. Risks RPN ≥ 100은 게이트 강제.
- **5 Whys**: 표면 추론 의심 시 / 사용자가 "근본 원인" 요청 시에만 추가.
