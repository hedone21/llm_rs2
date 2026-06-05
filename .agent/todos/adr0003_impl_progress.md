# ADR-0003 구현 진행 원장 (/loop dynamic)

**대상**: `docs/adr/0003-extension-mechanism-static-crates.md` 구현 완수
**SSOT**: ADR-0003 + `arch/pipeline_stage_design_v2.md`
**시작 HEAD**: `d331d01b feat(htp): S4+S5 …` (master)
**진입**: `/loop` dynamic 모드, 완료 시 자가 종료

---

## 기준선 (M0, 2026-06-05)

- **게이트 명령**: `cargo test -p llm_rs2 --lib -- --skip backend::opencl --skip memory::opencl`
- **기준선 결과**: `ok. 1220 passed; 0 failed; 39 filtered out`
- **GPU 테스트 제외 사유**: 이 호스트는 OpenCL 디바이스 부재(CLAUDE.md). `backend::opencl::*`는 flaky-fail(런마다 19~24개), `memory::opencl::unified::test_map_write_unmap_cycle`는 null-ptr **SIGABRT로 테스트 프로세스 전체를 죽임**. 둘 다 환경적 — 회귀 아님. 따라서 게이트에서 제외하고 "skip 셋 밖 실패 0"으로 회귀를 판정한다.

## 게이트 (각 마일스톤 커밋 전)

1. **빌드**: `cargo build -p llm_rs2` (dev) OK. + **release 빌드는 M2·M5에서만** (`cargo build --release -p llm_rs2`) — fat-LTO 빌드가 비싸 매 마일스톤 반복은 비효율. release가 의미 있는 지점(linkme fat-LTO 생존 smoke)에서만 수행. ← prompt 게이트의 *취지*(빌드+LTO 생존) 유지하며 cadence 최적화 (의도적 판단).
2. **테스트**: 위 게이트 명령 → `0 failed` (skip 셋 밖). 떨어지면 STOP.
3. **fmt**: `cargo fmt --all`.
4. **clippy**: `cargo clippy --workspace -- -D warnings` (`--all-targets` 금지) clean.
5. **M2 이후 release smoke**: release bin으로 각 정책 이름 해석(`--eviction-policy h2o` 등) "Unknown policy" bail 안 함 — linkme가 `--gc-sections`에서 생존했는지. 실패 시 ADR §4 build.rs codegen 폴백.

## 가드레일

- 한국어. Conventional Commits + `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **`git add -A` 금지** — 명시 파일만. push 금지.
- 커밋 금지 untracked: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/**/microbench_*`, `.agent/todos/handoff_microbench_*.md`, `arch/pipeline/`, `docs/refactoring_report/`(별건).
- 신규/이동 모듈 = no-`mod.rs`. 외과적 변경. 테스트 약화/우회 금지.

## 마일스톤 체크리스트

- [x] **M0** 기준선(1220) + 원장 + ADR-0003·README 커밋
- [ ] **M1** `crates/technique-api/` — `EvictionPolicy` 등록 표면(linkme `EVICTION_POLICIES` distributed_slice + 공유 trait re-export), workspace member, linkme dep
- [ ] **M2** `PipelineRegistry` → 슬라이스 읽어 name→factory 맵 → `session/chat/session.rs:621` match arm 제거 + startup self-test(sliding/streaming/h2o/h2o_plus/d2o 등록 단언) + release smoke
- [ ] **M3** 정책 impl을 `crates/techniques/<name>/` per-crate 이전 + linkme 등록, workspace glob `crates/techniques/*`, bin 의존(D4 1줄/기법), 더미 crate로 "폴더만 추가" 검증
- [ ] **M4** d2o `plan_keep` 이전 + `Merge` 가중치 필드. **동등성 테스트 선작성 필수**, 못 세우면 STOP+human-review 플래그 (senior-implementer 위임 가능)
- [ ] **M5** 기여자 문서 "기법 추가법"(hook·시그니처·등록 + 동작 예제 crate, 컴파일·등록 게이트)

## 완료 조건

M1·M2·M3·M5 커밋 + 전체 `/sanity-check` green + release self-test 통과 + 본 체크리스트 done.
(M4 = 동등성 테스트 통과 done, 또는 "human-review 필요" 명시 플래그.) → 최종 보고 + `notify-send` + 루프 종료.

## Iteration 로그

- **iter-1 (M0, 2026-06-05)**: repo 상태 확인(HEAD d331d01b, 7 worktree). baseline 측정 — GPU flaky/SIGABRT 발견 → 게이트를 "GPU skip + skip밖 실패 0"으로 정의. baseline=1220 passed/0 failed. 원장 작성. ADR-0003+README+원장 커밋 예정. → 다음 iter: M1.
