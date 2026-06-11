# Handoff: score accumulator 배선 + ADR-0006 Deferred 완주 → QCF_kv 설계 라운드

**작성**: 2026-06-12
**HEAD**: `4444bdc8 fix(qcf): GPU KV는 read_buffer로 V 읽기 강제 — rpcmem stale-cache QCF=0 해소` (+ 본 handoff/backlog 문서 커밋)
**브랜치**: master (worktree 없음, origin 미푸시 — 푸시는 사용자 지시 대기)
**다음 세션 진입 문장**: **"QCF_kv 설계 라운드 진행 — backlog 'QCF_kv 정규화 비대칭' 항목 기준"** (대안 트랙: "swap reversal 설계 라운드" / "host lib 테스트 위생")

---

## TL;DR

전 세션 handoff의 두 후보 트랙을 한 세션에 완주했다. **① score accumulator 배선**(Track A): 공유 cell 패턴(arch v2 §5.9)으로 생성→forward→QCF→eviction 전 구간 배선 완료, S25 트레이스로 체인 정상 입증 + 부산물로 사전존재 결함 3건(capability 정적 리스트·fixture relief 키·rpcmem stale V) 수정. **② ADR-0006 Deferred**(Track B): IntraForward/LayerImmediate hook forward slot 실배선 + S25 swap 3-mode 게이트 3/3 GREEN. 멈춘 이유 = `signal_memory_critical` known-fail 2건의 **최종 잔여 원인이 QCF_kv 수식 포화(3층 결함)로 재진단** — spec ENG-ALG-051 변경 + manager floor 재설정이 얽힌 설계 결정이라 사용자 결정 사항으로 backlog 안건화하고 경계에서 정지.

## 진행 상태 (검증 게이트 수치)

| 작업 | 게이트 | 커밋 |
|---|---|---|
| Architect 통합 설계 (공유 cell §5.9 + §5.6.3/§5.6.7 보강 + ADR-0006 §6) | spec triage 신규 INV 0건 | `905cfbcf` |
| Track B: hook cell 인프라 + forward slot 배선 (`TransformerModelForwardArgs.layer_boundary_hook` 신설) | host: lib 1347 신규 FAIL 0 · beta4 8/8 | `5862325c` |
| Track A: score cell 체인 (forward 장착 + QCF token_scores + eviction ScoreContext/reset) | host: 신규 유닛 6종 GREEN · spec 679 (FAIL 5 = 전부 사전존재, ceda1dfb worktree 대조 입증) | `3b241d11` + `1356011a` |
| capability 송출 동적화 (kv.evict_* 포함 12 actions) | S25 manager 수신 로그 실측 · thermal 정책별 차등 산출 확인 | `274472b6` |
| swap mode 가드 해제 | S25 3-mode RC=0 | `c0f1781f` |
| fixture relief 키 점 표기 정정 | 배포 fixture 반영 확인 | `b65690af` |
| rpcmem stale V-readback 강제 | S25: v0_abs 0→37.3, QCF 0→실값 | `4444bdc8` |
| **α-K frozen 재검증** | 3-dtype 생성 **byte-identical** + TBT Δ +0.53%/-0.11%/-0.39% (게이트 ≤+3%) | — |
| **swap 3-mode 게이트** | incremental: AB-6 frozen sig md5 정확 일치 + drain 7 tick 일치 / intra-forward·phase-aware·layer-immediate: 3/3 functional GREEN (marker + generated=128 + report 수신) | — |
| **최종 verify 매트릭스** | **28/30, 직전 매트릭스 대비 변동 0** (결과: `verify/results/20260612_022004_galaxy_s25_f16_q4/`) | — |

## 다음 작업 (택1 — 사용자 결정)

1. **(본선) QCF_kv 설계 라운드** — backlog `[P2] QCF_kv 정규화 비대칭 + estimator 우회 + manager floor 재설정` (안건 SSOT, Architect 분석 전문 수록). **사용자 결정 2개가 선행**: (a) estimator 우회 방향 — raw QCF 전송+manager측 환산 vs 엔진측 ΔPPL 환산(spec ENG-ALG-050 step 4 위반의 해소 방향), (b) manager QCF_FLOOR 재설정 승인(정책 의미 변경). 검증 게이트: `python verify/verify.py --device galaxy_s25 --model f16,q4 --scenario-filter signal_memory_critical` 2/2 GREEN.
2. swap reversal (RestoreDefaults F16 recall) — backlog `[P3]`, 설계 라운드 필요.
3. (위생, 비차단) backlog `[P2-chore]` host lib 테스트 위생 — POCL SIGABRT가 전체 스위트를 중단시켜 이번 세션도 `--skip backend::opencl --skip memory::opencl` 우회 필요했음.

## Landmines / 미해결

- **rpcmem `as_ptr()` ≠ cache-coherent**: DMA-BUF(--opencl-rpcmem)는 as_ptr이 항상 non-null이지만 GPU-written 데이터가 CPU 캐시 stale (UnifiedBuffer unmapped null-ptr와 **형제 클래스** — null 검사로는 못 잡는다). GPU KV를 CPU에서 읽는 모든 신규 코드는 `backend.read_buffer` D2H 필수 (`4444bdc8` 패턴).
- **v1(4월) signal_memory PASS는 우연**: fixture relief 키가 깨져(언더스코어 vs 점 표기) 전부 0이던 상태의 선택 역학에 의존했음이 git 포렌식으로 확정. v1 결과를 회귀 기준으로 삼을 때 이 클래스(설정 파일 키 불일치 = silent 0) 주의.
- **signal_memory_critical의 directive는 relief에 민감**: f16은 relief 정정 후 LayerSkip→SwitchHw로 변동, q4는 LayerSkip 유지 — 설계 라운드에서 floor 재설정 시 f16/q4가 갈리는 현 분포 참고 (`20260612_022004` 결과).
- **`attention_scores.cl failed to compile` WARN은 무해한 허위 단서**: score readback은 `kernel_attn_gen_half`(simple_ops.cl) legacy fallback으로 정상 작동 (S25 실측 ws.scores.sum=12.0). 이 WARN을 보고 score 경로를 의심하지 말 것.
- **hook lifetime = overwrite 시맨틱**: IntraForward hook cell은 plan 소진 후에도 잔존(클리어 경로 없음 — `should_dispatch` self-gate가 no-op 보장, v1 등가). 동시 활성은 commit 경로 가드가 차단. 클리어 경로를 추가하려면 weight_swap.rs install arm 주석의 결정 근거 먼저 읽을 것.
- **plan path는 score/hook active 시 우회**: score-based policy 또는 swap hook 활성 중엔 GPU plan 경로가 비활성 (model_forward.rs). happy path 무영향(frozen 입증)이나 score-active 장기 decode의 TBT는 plan-off 기준임을 인지.
- **stale 진입 문서**: `handoff_ab_tracks_complete_2026_06_11.md`의 진입 문장("score accumulator 배선 진행")은 본 세션에서 소화됨 — 본 문서가 정본.

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ ("QCF_kv 설계 라운드 진행" — backlog 안건이 SSOT)
- 왜 멈췄나? ✓ (잔여가 spec+manager 정책 의미를 바꾸는 설계 결정 = 사용자 결정 사항)
- 최대 landmine 표면화? ✓ (rpcmem stale V 클래스 + v1 PASS 우연성)
- 게이트가 수치/명령? ✓ (28/30, scenario-filter 명령, frozen Δ%)
- 길이 적정? ✓ (3층 결함 상세는 backlog 항목으로 위임)
