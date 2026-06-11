> **[SUPERSEDED 2026-06-12]** 본 문서의 진입 문장(score accumulator 배선 / ADR-0006 Deferred)은 2026-06-12 세션에서 **둘 다 소화됨**. 정본 진입 문서 = `handoff_score_wiring_qcf_round_2026_06_12.md`.

# Handoff: AB-0~6 전 트랙 완주 (AB-2 KIVI + AB-5 verify 재가동 종결) → 후속 트랙 선택

**작성**: 2026-06-11
**HEAD**: `3ea20868 docs(handoff): AB-5 종결 — S25 verify 매트릭스 28/30 + known-fail 2 + 수정 체인 기록`
**브랜치**: master (worktree 없음, **origin push 완료** `b5184d33..3ea20868` 26 커밋)
**다음 세션 진입 문장**: **"score accumulator 배선 진행 — handoff_ab_tracks_complete 기준"** (대안 트랙: "ADR-0006 Deferred 진행 — IntraForward hook 실배선부터")

---

## TL;DR

argus-bench AB 트랙 전체(AB-0~6)가 종결됐다. 이번 세션 = **AB-2**(KvQuantDynamic → KiviQuantStage, host+device GREEN) + **AB-5**(verify.py 원격 재가동, S25 매트릭스 **28/30 PASS + known-fail 2**) + 부산물 **QCF estimate 역방향 IPC 재배선**(β-7 제거 후 유일 미복구 잔여, arch v2 §5.8). 멈춘 이유 = 트랙 완주 — 다음은 사용자 선택: (a) backlog `[P2]` score accumulator 배선(known-fail 2건 해소, 범위 소), (b) ADR-0006 Deferred(IntraForward hook 실배선 등, 범위 대).

## 진행 상태 (검증 게이트 수치)

| 작업 | 게이트 | 커밋 |
|---|---|---|
| AB-2 설계+구현 (A안 handle Arc화, KvMutate, sticky last-applied) | host: lib 신규 FAIL 0(worktree 대조)·beta4 8/8·clippy clean / device: α-K sig 15/15 + tbt Δ≤+3%, KvQuantDynamic sig 3/3 IDENTICAL + marker 글자단위 + heartbeat kv_dtype=q4, KIVI 오라클 L2=0.000000, C12 e2e 흡수 | `fbcc46f6`→`f47543e8` (8) |
| AB-2 frozen baseline | `.agent/todos/frozen_baseline_ab2_kvquant_2026_06_11.md` (md5 `728ab92e…`, Decode 104.75 ms/tok) | — |
| QCF IPC 재배선 (§5.8) | dispatcher 직결 `compute_and_send_qcf` + report_tx, `LoopControl.request_qcf` 삭제, spec seq_095 8/8, S25 thermal GREEN 전환 실증 | `bf6230e8`+`226d154b`+`e267bd50` |
| AB-5 verify 재가동 | **15 시나리오 × f16,q4 = 28 PASS + 2 known-fail** — AB-2/4 신규 시나리오(kvquant 2종·partition 3종) 전부 PASS | `171fe98f`→`1aee5497` |
| 종결 기록 | `handoff_argus_bench_ab0_ab3_2026_06_05.md` §AB-5(수정 체인 상세) + `handoff_ab246_stage_entry_2026_06_11.md`(AB-2/4/6) + 메모리 `project_ab5_verify_rewire` | `3ea20868` |

## 다음 작업 (택1 — 사용자 결정)

1. **(권장 후보) score accumulator 배선** — backlog `[P2] argus_bench AttentionScoreAccumulator 배선 (AB-1 잔여)`. 구현: score-based eviction policy 구성 시 ModelForward `forward_into`의 `score_accumulator: None`을 실 accumulator로 → QCF estimates에 kv.evict_h2o/kv.merge_d2o 포함. **검증 게이트**: `python verify/verify.py --device galaxy_s25 --model f16,q4 --scenario-filter signal_memory_critical --skip-build --skip-deploy` → 2/2 GREEN (현 known-fail이 그대로 회귀 게이트). 주의: score 수집의 hot path 비용 — v1은 need_scores 조건부였음. α-K frozen 재검증 동반 필수.
2. **ADR-0006 Deferred** — IntraForward/LayerImmediate hook 실배선(forward slot greenfield, §5.6.3), PhaseAware device 검증, swap 역전(RestoreDefaults). 범위 대 — Architect 설계 선행.
3. (위생, 비차단) backlog `[P2-chore] host lib 테스트 위생` — γ-3 테스트 버그 2건 + POCL 환경 ~25종 + octal 버그.

## Landmines / 미해결

- **verify signal 경로 3결함은 수정 완료지만 패턴 기억**: ① remote_run_dir에 model key 부재 → 같은 pid 2번째 model이 stale `engine.rc` 0.0s 오탐(`1aee5497`이 spawn 전 rm -f) — "첫 모델만 PASS, 둘째만 FAIL" 패턴이면 이 클래스. ② rc-poll은 `test -f` 게이트 필수. ③ 주입 anchor 고정 sleep 금지(event-driven, Capability marker). `adb pull`은 mtime 미보존 — 로컬 mtime=pull 시각.
- **known-fail 2건의 정확한 경계**: `signal_memory_critical`은 엔진 정상(RequestQcf→QcfEstimate 1 action→policy LayerSkip 선택) — 원인은 scores 부재로 h2o/d2o estimate 미산출. **policy를 고치는 방향(빈 estimate=free 취급) 금지** — manager 영역이고 시나리오 의도(KvEvict)와 다름.
- **QCF estimates의 device 함정**: v2 zero-copy KV=UnifiedBuffer는 unmapped 시 `as_ptr()=null`(v1 USE_HOST_PTR와 다름) — `e267bd50`이 read-back fallback으로 해소. 신규 KV 액션 estimate 추가 시 같은 함정 주의.
- **thermal 시나리오는 functional_only** — policy가 `SwitchHw{cpu}`를 발행해 q4에서 ROUGE 0.297 의도적 발산(kvquant 선례). accuracy 게이트 복원하려면 q4 캘리브레이션 선행 필요(4월 v1 매트릭스는 f16만 돌았음 — q4 absolute 기준 부재).
- **AB-2 잔여(동결 범위 밖)**: 2/8bit·역전환(→16) 시나리오 미동결, Q6 dlopen TBT(실제 KIVI `.so` 필요).
- **stale 진입 문서 주의**: `handoff_argus_bench_ab0_ab3`의 AB-2/4/6 배선 계층 서술은 여전히 stale(β 이전) — 도메인 인벤토리만 참조, 배선은 `handoff_ab246_stage_entry` + arch v2 §5.5~5.8이 정본.

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ ("score accumulator 배선 진행")
- 왜 멈췄나? ✓ (트랙 완주, 다음은 사용자 택1)
- 최대 landmine 표면화? ✓ (signal 하네스 3결함 패턴 + policy 수정 금지 경계)
- 게이트가 수치/명령? ✓ (28/30, scenario-filter 명령, sig 15/15)
- 길이 적정? ✓ (상세는 종결 기록 2문서 + §5.8 + frozen baseline으로 위임)
